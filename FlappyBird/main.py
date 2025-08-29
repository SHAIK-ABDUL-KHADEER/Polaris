import math, random, time, threading, numpy as np, json
import pygame, sounddevice as sd
from dataclasses import dataclass
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
FPS = 60
GRAVITY = 0.38
FLAP_STRENGTH = -7.8
MAX_FALL_SPEED = 10
PIPE_GAP = 250
PIPE_DISTANCE = 330
PIPE_SPEED = 3.2
GROUND_HEIGHT = 120

# Colors
SKY_TOP = (75, 180, 255)
SKY_BOTTOM = (180, 230, 255)
PIPE_COLOR = (60, 200, 80)
PIPE_LIP = (40, 180, 60)
GROUND_COLOR = (222, 206, 160)
GROUND_DARK = (210, 190, 145)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GOLD = (250, 210, 70)
VOICE_BAR_BG = (50, 50, 50)
VOICE_BAR_FILL = (50, 220, 50)
VOICE_BAR_THRESHOLD = (220, 50, 50)
INFO_BG = (30, 30, 40)
INFO_PANEL_PADDING = 30

HIGHSCORE_FILE = Path("highscore.json")

# -----------------------------
# Voice trigger
# -----------------------------
class VoiceFlap:
    def __init__(self, threshold=0.02, cooldown=0.25):
        self.threshold = threshold
        self.cooldown = cooldown
        self._flap_flag = False
        self._lock = threading.Lock()
        self._last_trigger = 0
        self._levels = []
        self.stream = sd.InputStream(callback=self.audio_callback)
        self.stream.start()

    def audio_callback(self, indata, frames, time_info, status):
        volume = float(np.sqrt(np.mean(indata ** 2)))
        self._levels.append(volume)
        now = time.time()
        if volume > self.threshold and (now - self._last_trigger) > self.cooldown:
            with self._lock:
                self._flap_flag = True
                self._last_trigger = now

    def consume_flap(self):
        with self._lock:
            f = self._flap_flag
            self._flap_flag = False
            return f

    def current_level(self):
        if not self._levels:
            return 0
        return np.mean(self._levels[-20:])

# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Bird:
    x: float
    y: float
    vy: float = 0
    angle: float = 0
    frame: int = 0

    def rect(self):
        return pygame.Rect(int(self.x) - 18, int(self.y) - 12, 36, 24)

@dataclass
class Pipe:
    x: float
    top_h: int
    passed: bool = False

    @property
    def gap_top(self): return self.top_h
    @property
    def gap_bottom(self): return self.top_h + PIPE_GAP

    def rects(self):
        top_rect = pygame.Rect(int(self.x), 0, 70, self.gap_top)
        bottom_rect = pygame.Rect(int(self.x), self.gap_bottom, 70, SCREEN_HEIGHT - GROUND_HEIGHT - self.gap_bottom)
        return top_rect, bottom_rect

# -----------------------------
# Clouds (parallax)
# -----------------------------
clouds = []

def init_clouds(width, height, count=15):
    global clouds
    clouds = []
    for _ in range(count):
        x = random.randint(0, width // 2)
        y = random.randint(50, height // 2)
        w = random.randint(50, 100)
        h = w // 2
        speed = random.uniform(10, 30)
        layer = random.choice([0.5, 0.8, 1.0])
        clouds.append([x, y, w, h, speed, layer])

def draw_background(surface, width, height, dt):
    for i in range(height - GROUND_HEIGHT):
        t = i / (height - GROUND_HEIGHT)
        r = int(SKY_TOP[0] * (1 - t) + SKY_BOTTOM[0] * t)
        g = int(SKY_TOP[1] * (1 - t) + SKY_BOTTOM[1] * t)
        b = int(SKY_TOP[2] * (1 - t) + SKY_BOTTOM[2] * t)
        pygame.draw.line(surface, (r, g, b), (0, i), (width // 2, i))
    for c in clouds:
        c[0] -= c[4] * c[5] * dt
        if c[0] + c[2] < 0:
            c[0] = width // 2 + random.randint(0, width // 2)
            c[1] = random.randint(50, height // 2)
        pygame.draw.ellipse(surface, (255, 255, 255, 180), (c[0], c[1], c[2], c[3]))

# -----------------------------
# Drawing
# -----------------------------
def draw_ground(surface, offset, width, height):
    y0 = height - GROUND_HEIGHT
    pygame.draw.rect(surface, GROUND_COLOR, (0, y0, width // 2, GROUND_HEIGHT))
    for i in range(-1, width // 40 + 2):
        x = i * 40 - int(offset) % 40
        pygame.draw.rect(surface, GROUND_DARK, (x, y0 + 80, 30, 10), border_radius=5)
        pygame.draw.rect(surface, GROUND_DARK, (x + 15, y0 + 60, 30, 12), border_radius=6)

def draw_pipe(surface, pipe: Pipe):
    top_rect, bottom_rect = pipe.rects()
    pygame.draw.rect(surface, PIPE_COLOR, top_rect)
    pygame.draw.rect(surface, PIPE_COLOR, bottom_rect)
    lip_thick = 10
    pygame.draw.rect(surface, PIPE_LIP, (top_rect.x - 4, top_rect.bottom - lip_thick, top_rect.width + 8, lip_thick))
    pygame.draw.rect(surface, PIPE_LIP, (bottom_rect.x - 4, bottom_rect.y, bottom_rect.width + 8, lip_thick))

def draw_bird(surface, bird: Bird):
    bird.frame = (bird.frame + 1) % 45
    wing_phase = bird.frame // 15
    body_color = (255, 240, 140)
    outline = (120, 90, 20)
    eye_color = BLACK
    beak_color = GOLD
    angle = max(-25, min(60, -bird.vy * 2))
    bird.angle = angle
    body_surf = pygame.Surface((48, 36), pygame.SRCALPHA)
    pygame.draw.ellipse(body_surf, body_color, (6, 8, 32, 20))
    pygame.draw.ellipse(body_surf, outline, (6, 8, 32, 20), 2)
    pygame.draw.circle(body_surf, WHITE, (28, 16), 4)
    pygame.draw.circle(body_surf, eye_color, (29, 16), 2)
    pygame.draw.polygon(body_surf, beak_color, [(36, 18), (44, 16), (44, 20)])
    wing_color = (255, 210, 110)
    if wing_phase == 0: wing_rect = pygame.Rect(10, 10, 18, 8)
    elif wing_phase == 1: wing_rect = pygame.Rect(10, 14, 18, 8)
    else: wing_rect = pygame.Rect(10, 18, 18, 8)
    pygame.draw.ellipse(body_surf, wing_color, wing_rect)
    pygame.draw.ellipse(body_surf, outline, wing_rect, 2)
    rotated = pygame.transform.rotate(body_surf, angle)
    rect = rotated.get_rect(center=(int(bird.x), int(bird.y)))
    surface.blit(rotated, rect)

# -----------------------------
# Flap particles
# -----------------------------
particles = []

def add_flap_particles(bird):
    for _ in range(5):
        x = bird.x - random.randint(5, 10)
        y = bird.y + random.randint(-5, 5)
        radius = random.randint(2, 4)
        life = random.uniform(0.3, 0.6)
        particles.append([x, y, radius, life])

def draw_particles(surface, dt):
    for p in particles[:]:
        p[3] -= dt
        if p[3] <= 0:
            particles.remove(p)
            continue
        alpha = int(255 * (p[3] / 0.6))
        surf = pygame.Surface((p[2]*2, p[2]*2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (255, 255, 150, alpha), (p[2], p[2]), p[2])
        surface.blit(surf, (p[0]-p[2], p[1]-p[2]))

# -----------------------------
# Game logic
# -----------------------------
def spawn_pipe(prev_x):
    min_top = 80
    max_top = SCREEN_HEIGHT - GROUND_HEIGHT - 80 - PIPE_GAP
    top_h = random.randint(min_top, max_top)
    return Pipe(prev_x + PIPE_DISTANCE, top_h)

def check_collision(bird, pipes):
    bird_rect = bird.rect()
    if bird.y >= SCREEN_HEIGHT - GROUND_HEIGHT - 8: return True
    if bird.y <= 0: return True
    for p in pipes:
        tr, br = p.rects()
        if bird_rect.colliderect(tr) or bird_rect.colliderect(br): return True
    return False

# -----------------------------
# Highscore
# -----------------------------
def load_highscore():
    if HIGHSCORE_FILE.exists():
        with open(HIGHSCORE_FILE, "r") as f:
            data = json.load(f)
            return data.get("highscore", 0)
    return 0

def save_highscore(score):
    highscore = load_highscore()
    if score > highscore:
        with open(HIGHSCORE_FILE, "w") as f:
            json.dump({"highscore": score}, f)

# -----------------------------
# Info panel
# -----------------------------
def draw_info_panel(surface, width, height, font, big_font, bird, pipes, score, voice):
    panel_x = width // 2
    panel_w = width // 2
    pygame.draw.rect(surface, INFO_BG, (panel_x, 0, panel_w, height))

    title = big_font.render("Voice Flappy", True, WHITE)
    surface.blit(title, (panel_x + (panel_w - title.get_width()) // 2, 50))
    instructions = [
        "Sound-sensitive Flappy Bird!",
        "Make any sound or press SPACE/UP",
        "Avoid the pipes",
        "Score increases when passing pipes"
    ]
    for i, line in enumerate(instructions):
        surf = font.render(line, True, WHITE)
        surface.blit(surf, (panel_x + INFO_PANEL_PADDING, 150 + i*35))

    stats_box_h = 160
    stats_y = 150 + len(instructions)*35 + 20
    pygame.draw.rect(surface, (50,50,50), (panel_x + INFO_PANEL_PADDING-10, stats_y-10, panel_w-2*INFO_PANEL_PADDING+20, stats_box_h), border_radius=10)
    next_pipe_dist = min([p.x - bird.x for p in pipes if p.x > bird.x], default=0)
    highscore = load_highscore()
    stats_lines = [
        f"Score: {score}",
        f"Highscore: {highscore}",
        f"Bird velocity: {bird.vy:.2f}",
        f"Next pipe distance: {int(next_pipe_dist)}",
        f"Mic level: {voice.current_level():.3f}"
    ]
    for i, line in enumerate(stats_lines):
        surf = font.render(line, True, GOLD if "Highscore" in line else WHITE)
        surface.blit(surf, (panel_x + INFO_PANEL_PADDING, stats_y + i*30))

    draw_voice_bar(surface, voice, width, height, font, bottom=True)

# -----------------------------
# Voice bar
# -----------------------------
def draw_voice_bar(surface, voice, width, height, font, bottom=False):
    bar_w, bar_h = 200, 16
    x = width//2 + INFO_PANEL_PADDING
    y = height - 60 if bottom else height - 40
    pygame.draw.rect(surface, VOICE_BAR_BG, (x-2, y-2, bar_w+4, bar_h+4), border_radius=8)
    lvl = voice.current_level()
    fill = int(bar_w * min(1, lvl / 0.03))
    r = min(255, 50 + int(170 * lvl / 0.03))
    g = min(255, 220)
    b = min(255, 50)
    glow_color = (r, g, b)
    pygame.draw.rect(surface, glow_color, (x, y, fill, bar_h), border_radius=8)
    thresh_x = int(bar_w * (voice.threshold / 0.03))
    pygame.draw.line(surface, VOICE_BAR_THRESHOLD, (x+thresh_x, y-3), (x+thresh_x, y+bar_h+3), 3)
    tip = font.render("Mic", True, WHITE)
    surface.blit(tip, (x-4, y-30))

# -----------------------------
# Main game
# -----------------------------
def game_loop():
    global SCREEN_HEIGHT
    pygame.init()
    info = pygame.display.Info()
    SCREEN_WIDTH, SCREEN_HEIGHT = info.current_w, info.current_h
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Voice Flappy (sound sensitive)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Verdana", 28)
    big_font = pygame.font.SysFont("Verdana", 48, bold=True)
    voice = VoiceFlap(threshold=0.15)
    init_clouds(SCREEN_WIDTH, SCREEN_HEIGHT)

    def reset():
        bird = Bird(SCREEN_WIDTH*0.35/2, SCREEN_HEIGHT*0.45)
        bird.vy = 0
        pipes = [Pipe(SCREEN_WIDTH//2 + 120, random.randint(120, SCREEN_HEIGHT - GROUND_HEIGHT - 120 - PIPE_GAP))]
        while pipes[-1].x < SCREEN_WIDTH//2 + 3*PIPE_DISTANCE:
            pipes.append(spawn_pipe(pipes[-1].x))
        ground_offset = 0
        score = 0
        started = False
        dead = False
        return bird, pipes, ground_offset, score, started, dead

    bird, pipes, ground_offset, score, started, dead = reset()
    running = True
    while running:
        dt = clock.tick(FPS) / 1_000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    if not dead: bird.vy=FLAP_STRENGTH; started=True; add_flap_particles(bird)
                    else: bird, pipes, ground_offset, score, started, dead = reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if not dead: bird.vy=FLAP_STRENGTH; started=True; add_flap_particles(bird)
                    else: bird, pipes, ground_offset, score, started, dead = reset()

        if not dead and voice.consume_flap():
            bird.vy = FLAP_STRENGTH
            started = True
            add_flap_particles(bird)

        if started and not dead:
            bird.vy = min(MAX_FALL_SPEED, bird.vy + GRAVITY)
            bird.y += bird.vy
            for p in pipes: p.x -= PIPE_SPEED
            if pipes and pipes[0].x < -90: pipes.pop(0)
            if pipes[-1].x < SCREEN_WIDTH//2 + PIPE_DISTANCE: pipes.append(spawn_pipe(pipes[-1].x))
            for p in pipes:
                if not p.passed and p.x + 70 < bird.x: p.passed=True; score+=1
            if check_collision(bird, pipes):
                dead=True
                save_highscore(score)
            ground_offset += PIPE_SPEED

        # Draw
        draw_background(screen, SCREEN_WIDTH, SCREEN_HEIGHT, dt)
        for p in pipes: draw_pipe(screen, p)
        draw_ground(screen, ground_offset, SCREEN_WIDTH, SCREEN_HEIGHT)
        draw_particles(screen, dt)
        draw_bird(screen, bird)
        draw_info_panel(screen, SCREEN_WIDTH, SCREEN_HEIGHT, font, big_font, bird, pipes, score, voice)

        if not started and not dead:
            tip1 = font.render("Make a sound or press SPACE", True, WHITE)
            screen.blit(tip1, (SCREEN_WIDTH//4 - tip1.get_width()//2, SCREEN_HEIGHT//2 - 20))
        if dead:
            over = big_font.render("GAME OVER", True, WHITE)
            tip2 = font.render(f"Your score is {score}. Press SPACE to retry", True, WHITE)
            screen.blit(over, (SCREEN_WIDTH // 4 - over.get_width() // 2, SCREEN_HEIGHT // 2 - 60))
            screen.blit(tip2, (SCREEN_WIDTH // 4 - tip2.get_width() // 2, SCREEN_HEIGHT // 2))

        pygame.display.flip()

    pygame.quit()
    voice.stream.stop()
    voice.stream.close()

if __name__ == "__main__":
    game_loop()
