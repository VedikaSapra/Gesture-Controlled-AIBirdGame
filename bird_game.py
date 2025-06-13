import os, sys, random, cv2
import pygame
import mediapipe as mp

# ────────── CONSTANTS ──────────
WIDTH, HEIGHT = 600, 400
BIRD_W, BIRD_H, BIRD_X = 50, 40, 100
PIPE_W, PIPE_GAP, PIPE_SPEED = 60, 150, 5
SKY_COLOR = (135, 206, 235)
BROWN = (139, 69, 19)        # obstacle colour
WHITE, RED, BLACK = (255,255,255), (255,0,0), (0,0,0)

# ────────── INITIALISE DISPLAY FIRST ──────────
pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Bird Game – Head / Hand")
CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont("Arial", 32)

# ────────── SAFE IMAGE LOADER ──────────
def load_img(fname, target=None, alpha=False):
    if not os.path.exists(fname):
        print(f"[WARN] {fname} not found.")
        return None
    try:
        surf = pygame.image.load(fname)
        surf = surf.convert_alpha() if alpha else surf.convert()
        if target:
            surf = pygame.transform.smoothscale(surf, target)
        print(f"[OK] loaded {fname}")
        return surf
    except pygame.error as e:
        print(f"[WARN] {fname}: {e}")
        return None

# background & bird
SKY_IMG  = load_img("sky.png", (WIDTH, HEIGHT))
bird_img = load_img("bird.png", (BIRD_W, BIRD_H), alpha=True)
if not bird_img:
    bird_img = pygame.Surface((BIRD_W, BIRD_H)); bird_img.fill(RED)

# ────────── CHOOSE CONTROL MODE ──────────
def choose_mode():
    while True:
        WIN.fill(WHITE)
        msgs = ["Choose Control Mode",
                "1  -  Head / Face",
                "2  -  Hand / Wrist"]
        for i, txt in enumerate(msgs):
            surf = FONT.render(txt, True, BLACK)
            WIN.blit(surf, surf.get_rect(center=(WIDTH//2, HEIGHT//2-60+40*i)))
        pygame.display.update()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_1: return "head"
                if ev.key == pygame.K_2: return "hand"
        CLOCK.tick(30)

control_mode = choose_mode()
print("Mode selected:", control_mode)

# ────────── MEDIAPIPE ──────────
mp_hands, mp_face = mp.solutions.hands, mp.solutions.face_mesh
hand_det = mp_hands.Hands(max_num_hands=1,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7) if control_mode=="hand" else None
face_det = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) if control_mode=="head" else None
mp_draw = mp.solutions.drawing_utils
DOT = mp_draw.DrawingSpec((0,0,255), 2, 3)      # red dots
LINE = mp_draw.DrawingSpec((255,255,255), 2, 1)  # white lines

# ────────── CAMERA ──────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam"); sys.exit()

# ────────── HELPERS ──────────
def cv2surf(frame, scale=0.3):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
    return pygame.transform.smoothscale(surf, (int(w*scale), int(h*scale)))

def new_pipe():
    top_h = random.randint(50, HEIGHT-PIPE_GAP-50)
    top = pygame.Rect(WIDTH, 0, PIPE_W, top_h)
    bot = pygame.Rect(WIDTH, top_h+PIPE_GAP, PIPE_W, HEIGHT-(top_h+PIPE_GAP))
    return {"top": top, "bot": bot, "scored": False}

def collide(bird, pipes):
    if bird.top<=0 or bird.bottom>=HEIGHT: return True
    return any(bird.colliderect(p["top"]) or bird.colliderect(p["bot"]) for p in pipes)

def draw_screen(bird_rect, pipes, score, cam_surf, cam_rect):
    WIN.blit(SKY_IMG,(0,0)) if SKY_IMG else WIN.fill(SKY_COLOR)
    for p in pipes:
        pygame.draw.rect(WIN, BROWN, p["top"])
        pygame.draw.rect(WIN, BROWN, p["bot"])
    WIN.blit(bird_img, (bird_rect.x, bird_rect.y))
    WIN.blit(cam_surf, cam_rect); pygame.draw.rect(WIN, BLACK, cam_rect, 2)
    WIN.blit(FONT.render(f"Score: {score}", True, BLACK), (10,10))
    pygame.display.update()

def game_over(score):
    WIN.fill(WHITE)
    msgs=["Game Over!", f"Score: {score}", "R = Restart | Q = Quit"]
    for i,t in enumerate(msgs):
        surf=FONT.render(t,True, RED if i==0 else BLACK)
        WIN.blit(surf, surf.get_rect(center=(WIDTH//2, HEIGHT//2-50+40*i)))
    pygame.display.update()
    while True:
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: return False
            if ev.type==pygame.KEYDOWN:
                if ev.key==pygame.K_q: return False
                if ev.key==pygame.K_r: return True

# ────────── MAIN ROUND ──────────
def play_round():
    bird_y=smooth_y=HEIGHT//2
    pipes, score=[],0
    alpha=0.25
    while True:
        CLOCK.tick(30)
        ok, frame = cap.read(); 
        if not ok: continue
        frame=cv2.flip(frame,1); rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if control_mode=="hand":
            res=hand_det.process(rgb)
            if res.multi_hand_landmarks:
                lm=res.multi_hand_landmarks[0].landmark
                mp_draw.draw_landmarks(frame,res.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,DOT,LINE)   # draw only for hand
                y_vals=[lm[i].y for i in [0,8,12,16,20]]  # wrist+tips
                target=int(sum(y_vals)/len(y_vals)*HEIGHT)-BIRD_H//2
                smooth_y=int(alpha*target+(1-alpha)*smooth_y); bird_y=smooth_y
        else:                                             # head
            res=face_det.process(rgb)
            if res.multi_face_landmarks:
                nose=res.multi_face_landmarks[0].landmark[1]
                target=int(nose.y*HEIGHT)-BIRD_H//2
                smooth_y=int(alpha*target+(1-alpha)*smooth_y); bird_y=smooth_y
                # (No landmark drawing for head)

        bird_y=max(0,min(HEIGHT-BIRD_H,bird_y))
        cam_surf=cv2surf(frame); cam_rect=cam_surf.get_rect(bottomright=(WIDTH-10,HEIGHT-10))

        if not pipes or pipes[-1]["top"].x < WIDTH-300: pipes.append(new_pipe())
        for p in pipes:
            p["top"].x -= PIPE_SPEED; p["bot"].x -= PIPE_SPEED
            if not p["scored"] and p["top"].x+PIPE_W < BIRD_X:
                score+=1; p["scored"]=True
        pipes=[p for p in pipes if p["top"].x+PIPE_W>0]

        bird_rect=pygame.Rect(BIRD_X,bird_y,BIRD_W,BIRD_H)
        if collide(bird_rect,pipes): return score
        draw_screen(bird_rect,pipes,score,cam_surf,cam_rect)
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: return None

# ────────── MAIN LOOP ──────────
try:
    while True:
        res=play_round()
        if res is None or not game_over(res): break
except Exception:
    import traceback; traceback.print_exc()
finally:
    cap.release(); pygame.quit(); sys.exit()
