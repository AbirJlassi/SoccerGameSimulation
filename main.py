import sys
print(sys.executable)



import random
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import matplotlib.animation as animation
from matplotlib.widgets import Button
import matplotlib.colors as mcolors

# Define constants
TEAM_A_COLOR = 'red'
TEAM_B_COLOR = 'blue'
BALL_COLOR = 'yellow'
FIELD_WIDTH = 100
FIELD_HEIGHT = 60
GOAL_WIDTH = 10
GOAL_AREA = 15  # distance from each goal where players can shoot
PLAYER_SPEED = 1
BALL_SPEED = 2
MATCH_LENGTH = 500  # simulation steps
GOAL_ANIMATION_FRAMES = 30  # Number of frames for goal celebration animation

class Player(Agent):
    """Player agent that moves towards the ball and attempts to score goals."""
    
    def __init__(self, unique_id, model, team, x, y, role):
        super().__init__(unique_id, model)
        self.team = team  # either 'A' or 'B'
        self.pos = (x, y)
        self.role = role  # 'attacker', 'midfielder', or 'defender'
        self.has_ball = False
        self.speed = PLAYER_SPEED
        self.celebrating = False  # Flag to indicate if player is celebrating
        
    def move_towards(self, target_pos):
        """Move towards a target position."""
        # Skip movement if celebrating
        if self.celebrating:
            return
            
        dx = 0
        dy = 0
        
        if target_pos[0] > self.pos[0]:
            dx = min(self.speed, target_pos[0] - self.pos[0])
        elif target_pos[0] < self.pos[0]:
            dx = max(-self.speed, target_pos[0] - self.pos[0])
            
        if target_pos[1] > self.pos[1]:
            dy = min(self.speed, target_pos[1] - self.pos[1])
        elif target_pos[1] < self.pos[1]:
            dy = max(-self.speed, target_pos[1] - self.pos[1])
            
        new_pos = (self.pos[0] + dx, self.pos[1] + dy)
        
        # Check if the new position is within the field
        if 0 <= new_pos[0] < FIELD_WIDTH and 0 <= new_pos[1] < FIELD_HEIGHT:
            self.pos = new_pos
            if self.has_ball:
                self.model.ball.pos = new_pos
    
    def get_target_position(self):
        """Determine where the player should move based on role and team."""
        ball_pos = self.model.ball.pos
        
        # Define base positions by role and team
        if self.team == 'A':  # Team A attacks from left to right
            if self.role == 'attacker':
                base_x = min(75, ball_pos[0] + 10)
            elif self.role == 'midfielder':
                base_x = ball_pos[0]
            else:  # defender
                base_x = max(25, ball_pos[0] - 20)
        else:  # Team B attacks from right to left
            if self.role == 'attacker':
                base_x = max(25, ball_pos[0] - 10)
            elif self.role == 'midfielder':
                base_x = ball_pos[0]
            else:  # defender
                base_x = min(75, ball_pos[0] + 20)
        
        # If player has the ball, move towards opponent's goal
        if self.has_ball:
            if self.team == 'A':
                return (FIELD_WIDTH - 5, FIELD_HEIGHT / 2)
            else:
                return (5, FIELD_HEIGHT / 2)
        
        # If no player has the ball, go for it
        if not any(player.has_ball for player in self.model.schedule.agents if isinstance(player, Player)):
            return ball_pos
        
        # Otherwise, maintain formation
        return (base_x, ball_pos[1] + random.uniform(-10, 10))
    
    def attempt_to_score(self):
        """Try to score a goal if in range of the opponent's goal."""
        if not self.has_ball:
            return
            
        goal_chance = 0
        
        if self.team == 'A' and self.pos[0] >= FIELD_WIDTH - GOAL_AREA:
            # Check if in front of goal
            if (FIELD_HEIGHT / 2 - GOAL_WIDTH / 2) <= self.pos[1] <= (FIELD_HEIGHT / 2 + GOAL_WIDTH / 2):
                goal_chance = 0.7  # High chance if directly in front
            else:
                goal_chance = 0.3  # Lower chance if from an angle
                
            if random.random() < goal_chance:
                self.model.score('A')
                self.has_ball = False
                self.model.goal_scored = True
                self.model.goal_animation_frames = GOAL_ANIMATION_FRAMES
                self.model.scoring_team = 'A'
                self.model.start_celebration('A')
                # Ball will be reset after animation completes
                
        elif self.team == 'B' and self.pos[0] <= GOAL_AREA:
            # Check if in front of goal
            if (FIELD_HEIGHT / 2 - GOAL_WIDTH / 2) <= self.pos[1] <= (FIELD_HEIGHT / 2 + GOAL_WIDTH / 2):
                goal_chance = 0.7
            else:
                goal_chance = 0.3
                
            if random.random() < goal_chance:
                self.model.score('B')
                self.has_ball = False
                self.model.goal_scored = True
                self.model.goal_animation_frames = GOAL_ANIMATION_FRAMES
                self.model.scoring_team = 'B'
                self.model.start_celebration('B')
                # Ball will be reset after animation completes
    
    def try_to_get_ball(self):
        """Attempt to take possession of the ball."""
        if self.celebrating:
            return
            
        ball_pos = self.model.ball.pos
        
        # If the player is close to the ball and no one has it
        if (abs(self.pos[0] - ball_pos[0]) <= 1 and 
            abs(self.pos[1] - ball_pos[1]) <= 1 and 
            not any(player.has_ball for player in self.model.schedule.agents if isinstance(player, Player))):
            
            nearby_players = [
                agent for agent in self.model.schedule.agents
                if isinstance(agent, Player) and 
                abs(agent.pos[0] - ball_pos[0]) <= 1 and 
                abs(agent.pos[1] - ball_pos[1]) <= 1
            ]
            
            # If there are other players nearby, randomize who gets the ball
            if len(nearby_players) > 1:
                lucky_player = random.choice(nearby_players)
                if lucky_player.unique_id == self.unique_id:
                    self.has_ball = True
            else:
                self.has_ball = True
                
        # Players can steal the ball from opponents
        for player in self.model.schedule.agents:
            if (isinstance(player, Player) and 
                player.has_ball and 
                player.team != self.team and
                abs(self.pos[0] - player.pos[0]) <= 1 and 
                abs(self.pos[1] - player.pos[1]) <= 1):
                
                # 30% chance to steal the ball
                if random.random() < 0.3:
                    player.has_ball = False
                    self.has_ball = True
    
    def celebrate(self):
        """Perform celebration movement if player's team just scored."""
        if not self.celebrating:
            return
            
        # Random jumping movement for celebration
        dx = random.uniform(-2, 2)
        dy = random.uniform(-2, 2)
        
        new_pos = (self.pos[0] + dx, self.pos[1] + dy)
        
        # Keep within field boundaries
        new_pos = (
            max(2, min(FIELD_WIDTH - 2, new_pos[0])),
            max(2, min(FIELD_HEIGHT - 2, new_pos[1]))
        )
        
        self.pos = new_pos
    
    def step(self):
        # If celebrating, just do celebration movement
        if self.celebrating:
            self.celebrate()
            return
            
        # Try to get the ball if don't have it
        self.try_to_get_ball()
        
        # Get the target position based on role and situation
        target_pos = self.get_target_position()
        
        # Move towards the target
        self.move_towards(target_pos)
        
        # If the player has the ball, try to score
        if self.has_ball:
            self.attempt_to_score()

class Ball(Agent):
    """Ball agent that follows players or moves randomly."""
    
    def __init__(self, unique_id, model, x, y):
        super().__init__(unique_id, model)
        self.pos = (x, y)
        self.velocity = (0, 0)
        self.rotation = 0  # Rotation angle for the ball
    
    def step(self):
        # Skip movement if a goal was just scored
        if self.model.goal_scored:
            # Move the ball toward the goal net in animation
            if self.model.scoring_team == 'A':
                goal_pos = (FIELD_WIDTH + 1, FIELD_HEIGHT / 2)
            else:
                goal_pos = (-1, FIELD_HEIGHT / 2)
                
            # Move ball toward goal
            dx = (goal_pos[0] - self.pos[0]) * 0.2
            dy = (goal_pos[1] - self.pos[1]) * 0.2
            
            self.pos = (self.pos[0] + dx, self.pos[1] + dy)
            
            # Update rotation for spinning effect
            self.rotation += 15
            return
            
        # If no player has the ball, move slightly randomly
        if not any(player.has_ball for player in self.model.schedule.agents if isinstance(player, Player)):
            # Apply some friction to slow down
            self.velocity = (self.velocity[0] * 0.9, self.velocity[1] * 0.9)
            
            # Random small movement
            if random.random() < 0.3:  # Only sometimes change direction
                self.velocity = (
                    self.velocity[0] + random.uniform(-0.5, 0.5),
                    self.velocity[1] + random.uniform(-0.5, 0.5)
                )
            
            # Apply velocity limits
            max_speed = BALL_SPEED
            magnitude = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
            if magnitude > max_speed:
                self.velocity = (self.velocity[0] * max_speed / magnitude, 
                                 self.velocity[1] * max_speed / magnitude)
            
            # Update position
            new_x = self.pos[0] + self.velocity[0]
            new_y = self.pos[1] + self.velocity[1]
            
            # Bounce off boundaries
            if new_x <= 0 or new_x >= FIELD_WIDTH:
                self.velocity = (-self.velocity[0], self.velocity[1])
                new_x = max(0, min(FIELD_WIDTH, new_x))
            if new_y <= 0 or new_y >= FIELD_HEIGHT:
                self.velocity = (self.velocity[0], -self.velocity[1])
                new_y = max(0, min(FIELD_HEIGHT, new_y))
                
            self.pos = (new_x, new_y)
            
            # Update rotation for rolling effect
            self.rotation += 5
            if self.rotation >= 360:
                self.rotation = 0

class FootballModel(Model):
    """Model for the football/soccer game."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        
        # Create two teams of players (5 players per team for simplicity)
        self.create_teams()
        
        # Create the ball and add it to the model
        self.ball = Ball(999, self, width // 2, height // 2)  # Ball starts in the center
        self.schedule.add(self.ball)
        
        # Initialize score and game time
        self.team_a_score = 0
        self.team_b_score = 0
        self.time = 0
        self.game_over = False
        self.match_started = False
        self.result_message = ""
        
        
        
        
        # Goal animation variables
        self.goal_scored = False
        self.goal_animation_frames = 0
        self.scoring_team = None
        self.goal_flash_color = 'white'
        
        # Setup the data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Team A Score": lambda m: m.team_a_score,
                "Team B Score": lambda m: m.team_b_score,
                "Time": lambda m: m.time
            }
        )
    
    def create_teams(self):
        """Create the teams with different roles."""
        # Team A (Red) - Left to Right
        team_a = [
            # Defenders
            {'id': 1, 'team': 'A', 'role': 'defender', 'pos': (15, 20)},
            {'id': 2, 'team': 'A', 'role': 'defender', 'pos': (15, 40)},
            # Midfielders
            {'id': 3, 'team': 'A', 'role': 'midfielder', 'pos': (35, 20)},
            {'id': 4, 'team': 'A', 'role': 'midfielder', 'pos': (35, 40)},
            # Attackers
            {'id': 5, 'team': 'A', 'role': 'attacker', 'pos': (60, 30)},
        ]
        
        # Team B (Blue) - Right to Left
        team_b = [
            # Defenders
            {'id': 6, 'team': 'B', 'role': 'defender', 'pos': (85, 20)},
            {'id': 7, 'team': 'B', 'role': 'defender', 'pos': (85, 40)},
            # Midfielders
            {'id': 8, 'team': 'B', 'role': 'midfielder', 'pos': (65, 20)},
            {'id': 9, 'team': 'B', 'role': 'midfielder', 'pos': (65, 40)},
            # Attackers
            {'id': 10, 'team': 'B', 'role': 'attacker', 'pos': (40, 30)},
        ]
        
        # Add all players to the model
        for player_data in team_a + team_b:
            player = Player(
                player_data['id'], 
                self, 
                player_data['team'], 
                player_data['pos'][0], 
                player_data['pos'][1], 
                player_data['role']
            )
            self.schedule.add(player)
    
    def start_celebration(self, team):
        """Start celebration animation for the scoring team."""
        for agent in self.schedule.agents:
            if isinstance(agent, Player):
                # Only players of the scoring team celebrate
                if agent.team == team:
                    agent.celebrating = True
    
    def end_celebration(self):
        """End celebration and reset positions."""
        for agent in self.schedule.agents:
            if isinstance(agent, Player):
                agent.celebrating = False
                
        self.reset_after_goal()
        
    def reset_after_goal(self):
        """Reset ball and players after a goal is scored."""
        # Reset ball to center with no velocity
        self.ball.pos = (self.width // 2, self.height // 2)
        self.ball.velocity = (0, 0)
        self.ball.rotation = 0
        
        # Reset player positions somewhat
        for agent in self.schedule.agents:
            if isinstance(agent, Player):
                if agent.team == 'A':
                    base_x = self.width // 4
                else:
                    base_x = 3 * self.width // 4
                    
                # Add some random variation to positioning
                agent.pos = (
                    base_x + random.randint(-10, 10),
                    agent.pos[1] + random.randint(-5, 5)
                )
                
                # Keep players in bounds
                agent.pos = (
                    max(5, min(self.width - 5, agent.pos[0])),
                    max(5, min(self.height - 5, agent.pos[1]))
                )
                
                agent.has_ball = False
                
        # Reset goal animation flags
        self.goal_scored = False
        self.goal_animation_frames = 0
        self.scoring_team = None
    
    def score(self, team):
        """Update the score based on which team scores."""
        if team == 'A':
            self.team_a_score += 1
        elif team == 'B':
            self.team_b_score += 1
            
        # Log the score
        print(f"GOAL! Team {team} scores! Current score: A {self.team_a_score} - {self.team_b_score} B")

    def step(self):
        """Advance the model by one step."""
        if not self.match_started or self.game_over:
            return
            
        # Handle goal animation
        if self.goal_scored:
            self.goal_animation_frames -= 1
            
            # Switch flash color for goal celebration
            if self.goal_animation_frames % 5 == 0:
                if self.goal_flash_color == 'white':
                    self.goal_flash_color = TEAM_A_COLOR if self.scoring_team == 'A' else TEAM_B_COLOR
                else:
                    self.goal_flash_color = 'white'
                    
            # End goal celebration when animation frames are done
            if self.goal_animation_frames <= 0:
                self.end_celebration()
                
            # During goal animation, we only want the ball to move toward the goal
            self.ball.step()
            return
            
        self.time += 1
        self.datacollector.collect(self)
        self.schedule.step()
        
        # Check if the match is over
        if self.time >= MATCH_LENGTH:
            self.game_over = True
            self.determine_winner()
    
    def determine_winner(self):
        """Determine the winner and set the result message."""
        if self.team_a_score > self.team_b_score:
            self.result_message = f"FULL TIME: Team A {self.team_a_score} - {self.team_b_score} Team B\nTeam A (Red) wins the match!"
        elif self.team_b_score > self.team_a_score:
            self.result_message = f"FULL TIME: Team A {self.team_a_score} - {self.team_b_score} Team B\nTeam B (Blue) wins the match!"
        else:
            self.result_message = f"FULL TIME: Team A {self.team_a_score} - {self.team_b_score} Team B\nThe match ends in a draw!"
        
        print("\n" + "="*50)
        print(self.result_message)
        print("="*50)
        
    def get_score(self):
        return self.team_a_score, self.team_b_score
    
    def start_match(self):
        """Start the football match."""
        self.match_started = True

class FootballVisualization:
    """Visualize the football model with matplotlib."""
    
    def __init__(self, model):
        self.model = model
        
        # Setup the figure with subplots
        # Main plot for football field, small area at bottom for buttons
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_axes([0.05, 0.1, 0.9, 0.85])  # Main axes for the field
        self.button_ax = self.fig.add_axes([0.4, 0.02, 0.2, 0.05])  # Axes for the start button
        
        # Create the start button
        self.start_button = Button(self.button_ax, 'Start Match', color='lightgreen', hovercolor='green')
        self.start_button.on_clicked(self.start_match)
        
        self.setup_field()
        self.anim = None
        self.frame_counter = 0
        self.show_initial_state()
    
    def setup_field(self):
        """Setup the football field layout."""
        self.ax.set_xlim(-5, FIELD_WIDTH + 5)  # Extended to show goals
        self.ax.set_ylim(-5, FIELD_HEIGHT + 5)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#538032')  # green grass color
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Football Match Simulation", fontsize=16)
    
    def draw_field_markings(self):
        """Draw the football field markings."""
        # Field outline
        self.ax.plot([0, FIELD_WIDTH], [0, 0], color="white", linewidth=2)  # Bottom line
        self.ax.plot([0, FIELD_WIDTH], [FIELD_HEIGHT, FIELD_HEIGHT], color="white", linewidth=2)  # Top line
        self.ax.plot([0, 0], [0, FIELD_HEIGHT], color="white", linewidth=2)  # Left line
        self.ax.plot([FIELD_WIDTH, FIELD_WIDTH], [0, FIELD_HEIGHT], color="white", linewidth=2)  # Right line

        # Center line and circle
        self.ax.plot([FIELD_WIDTH/2, FIELD_WIDTH/2], [0, FIELD_HEIGHT], color="white", linewidth=2)
        center_circle = plt.Circle((FIELD_WIDTH/2, FIELD_HEIGHT/2), 10, color='white', fill=False, linewidth=2)
        self.ax.add_patch(center_circle)
        center_dot = plt.Circle((FIELD_WIDTH/2, FIELD_HEIGHT/2), 1, color='white')
        self.ax.add_patch(center_dot)

        # Goal areas
        # Left goal area
        self.ax.plot([0, 15], [15, 15], color="white", linewidth=2)
        self.ax.plot([15, 15], [15, FIELD_HEIGHT-15], color="white", linewidth=2)
        self.ax.plot([0, 15], [FIELD_HEIGHT-15, FIELD_HEIGHT-15], color="white", linewidth=2)
        
        # Right goal area
        self.ax.plot([FIELD_WIDTH-15, FIELD_WIDTH], [15, 15], color="white", linewidth=2)
        self.ax.plot([FIELD_WIDTH-15, FIELD_WIDTH-15], [15, FIELD_HEIGHT-15], color="white", linewidth=2)
        self.ax.plot([FIELD_WIDTH-15, FIELD_WIDTH], [FIELD_HEIGHT-15, FIELD_HEIGHT-15], color="white", linewidth=2)
        
        # Goals
        goal_height = GOAL_WIDTH
        goal_y_center = FIELD_HEIGHT / 2
        
        # Left goal
        left_goal = Rectangle((-3, goal_y_center - goal_height/2), 3, goal_height, 
                              linewidth=2, edgecolor='white', facecolor='lightgray', alpha=0.3)
        self.ax.add_patch(left_goal)
        
        # Goal net lines (left)
        net_spacing = goal_height / 5
        for i in range(6):
            y = goal_y_center - goal_height/2 + i * net_spacing
            self.ax.plot([-3, 0], [y, y], color="white", linewidth=1, alpha=0.5)
        for i in range(4):
            x = -3 + i * 1
            self.ax.plot([x, x], [goal_y_center - goal_height/2, goal_y_center + goal_height/2], 
                        color="white", linewidth=1, alpha=0.5)
        
        # Right goal
        right_goal = Rectangle((FIELD_WIDTH, goal_y_center - goal_height/2), 3, goal_height, 
                               linewidth=2, edgecolor='white', facecolor='lightgray', alpha=0.3)
        self.ax.add_patch(right_goal)
        
        # Goal net lines (right)
        for i in range(6):
            y = goal_y_center - goal_height/2 + i * net_spacing
            self.ax.plot([FIELD_WIDTH, FIELD_WIDTH+3], [y, y], color="white", linewidth=1, alpha=0.5)
        for i in range(4):
            x = FIELD_WIDTH + i * 1
            self.ax.plot([x, x], [goal_y_center - goal_height/2, goal_y_center + goal_height/2], 
                        color="white", linewidth=1, alpha=0.5)
    
    def draw_agents(self):
        """Draw the players and ball."""
        for agent in self.model.schedule.agents:
            if isinstance(agent, Ball):
                # Draw a more detailed ball with pentagon-hexagon pattern
                self.draw_ball(agent)
            
            if isinstance(agent, Player):
                color = TEAM_A_COLOR if agent.team == 'A' else TEAM_B_COLOR
                # Make the player with the ball appear larger
                size = 2.5 if agent.has_ball else 2
                player = Circle(agent.pos, size, color=color, alpha=0.8, edgecolor='black', linewidth=1)
                self.ax.add_patch(player)
                
                # Display player number
                self.ax.text(agent.pos[0], agent.pos[1], str(agent.unique_id), 
                            fontsize=8, ha='center', va='center', color='white')
                
    def draw_ball(self, ball):
        """Draw a more detailed soccer ball."""
        # Base white ball
        ball_circle = Circle(ball.pos, 2, color='white', edgecolor='black', linewidth=1)
        self.ax.add_patch(ball_circle)
        
        # Calculate pentagon points for a soccer ball appearance
        # Adjust based on ball rotation
        angle = ball.rotation * (np.pi / 180)  # Convert degrees to radians
        
        # Create pentagon pattern
        pentagon_points = []
        for i in range(5):
            theta = angle + i * 2 * np.pi / 5
            r = 1.3  # Size of the pattern
            x = ball.pos[0] + r * np.cos(theta)
            y = ball.pos[1] + r * np.sin(theta)
            pentagon_points.append((x, y))
        
        # Add the pentagon patches to the ball
        pentagon = Polygon(pentagon_points, closed=True, color='black', alpha=0.8)
        self.ax.add_patch(pentagon)
        
        # Add a shine effect
        shine_pos = (ball.pos[0] + 0.7, ball.pos[1] + 0.7)
        shine = Circle(shine_pos, 0.5, color='white', alpha=0.5)
        self.ax.add_patch(shine)
    
    def show_initial_state(self):
        """Show the initial state of the field before the match starts."""
        self.draw_field_markings()
        self.draw_agents()
        
        # Show team names and instructions
        self.ax.text(FIELD_WIDTH/4, FIELD_HEIGHT + 2, "Team A (Red)", 
                    color=TEAM_A_COLOR, fontsize=14, ha='center')
        self.ax.text(3*FIELD_WIDTH/4, FIELD_HEIGHT + 2, "Team B (Blue)", 
                    color=TEAM_B_COLOR, fontsize=14, ha='center')
        self.ax.text(FIELD_WIDTH/2, -5, "Click 'Start Match' to begin", 
                    color='black', fontsize=12, ha='center')
    
    def display_score(self):
        """Display the current score and time."""
        team_a_score, team_b_score = self.model.get_score()
        score_text = f"Team A (Red): {team_a_score} | Team B (Blue): {team_b_score} | Time: {self.model.time}/{MATCH_LENGTH}"
        
        # Background rectangle for score text
        score_bg = Rectangle((FIELD_WIDTH/2 - 30, 3), 60, 10, 
                            facecolor='black', alpha=0.7, edgecolor=None)
        self.ax.add_patch(score_bg)
        
        # Score text
        self.ax.text(FIELD_WIDTH/2, 8, score_text, 
                    ha='center', fontsize=12, color='white')
    
    # Complete the display_goal_animation function that was cut off
    def display_goal_animation(self):
        """Display the goal celebration animation."""
        if not self.model.goal_scored:
            return
            
        # Flash the background with team color
        flash_alpha = 0.3 if self.model.goal_animation_frames % 10 < 5 else 0.1
        flash_rect = Rectangle((-5, -5), FIELD_WIDTH + 10, FIELD_HEIGHT + 10, 
                              facecolor=self.model.goal_flash_color, alpha=flash_alpha, edgecolor=None)
        self.ax.add_patch(flash_rect)
        
        # Show GOAL! text
        goal_text = "GOAL!"
        text_size = 24 + (GOAL_ANIMATION_FRAMES - self.model.goal_animation_frames) // 2
        text_size = min(text_size, 36)  # Cap at reasonable size
        
        self.ax.text(FIELD_WIDTH/2, FIELD_HEIGHT/2, goal_text, 
                    ha='center', va='center', fontsize=text_size, 
                    color=self.model.goal_flash_color, 
                    weight='bold', alpha=0.8)
        
        # Show who scored
        team_text = f"Team {'A (Red)' if self.model.scoring_team == 'A' else 'B (Blue)'}"
        self.ax.text(FIELD_WIDTH/2, FIELD_HEIGHT/2 - 10, team_text, 
                    ha='center', va='center', fontsize=18, 
                    color=self.model.goal_flash_color, 
                    weight='bold', alpha=0.8)
    
    def display_match_result(self):
        """Display the final match result when the game is over."""
        if not self.model.game_over:
            return
            
        # Create a semi-transparent overlay
        result_bg = Rectangle((-5, -5), FIELD_WIDTH + 10, FIELD_HEIGHT + 10, 
                             facecolor='black', alpha=0.7, edgecolor=None)
        self.ax.add_patch(result_bg)
        
        # Display the result message
        lines = self.model.result_message.split('\n')
        for i, line in enumerate(lines):
            self.ax.text(FIELD_WIDTH/2, FIELD_HEIGHT/2 - (i-1) * 10, line, 
                        ha='center', va='center', fontsize=20, 
                        color='white', weight='bold')
    
    def animate(self, i):
        """Animate the football simulation."""
        self.ax.clear()
        self.draw_field_markings()
        
        # Step the model forward
        if self.model.match_started and not self.model.game_over:
            self.model.step()
        
        # Draw all agents
        self.draw_agents()
        
        # Display the score
        self.display_score()
        
        # Display goal animation if needed
        self.display_goal_animation()
        
        # Display match result if game is over
        self.display_match_result()
        
        # Fix for premature match ending - check if match should be over
        # This ensures the match continues until MATCH_LENGTH is reached
        if self.model.match_started and self.model.time >= MATCH_LENGTH and not self.model.game_over:
            self.model.game_over = True
            self.model.determine_winner()
        
        return self.ax
    
    def start_match(self, event):
        """Start the match when the button is clicked."""
        # Disable the button after it's clicked
        self.start_button.label.set_text('Match Started')
        self.start_button.color = 'gray'
        self.start_button.hovercolor = 'gray'
        
        # Start the model
        self.model.start_match()
        
        # Start the animation if not already running
        if self.anim is None:
            self.anim = animation.FuncAnimation(
                self.fig, self.animate, interval=50, blit=False, save_count=MATCH_LENGTH + 100)
            plt.draw()

# Add the main function to run the simulation
def run_football_simulation():
    # Create the model and visualization
    model = FootballModel(FIELD_WIDTH, FIELD_HEIGHT)
    viz = FootballVisualization(model)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    run_football_simulation()