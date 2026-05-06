import torch
torch.set_float32_matmul_precision("high")
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
import shapiq 
import src

if __name__ == "__main__":
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    input_text = "black dog next to a yellow hydrant"
    input_image = Image.open("assets/dog_and_hydrant.png")
    
    game = src.game_huggingface.VisionLanguageGame(
    model=model,
    processor=processor,
    input_image=input_image,
    input_text=input_text,
    batch_size=64
    )
    n_players_image = game.n_players_image
    n_players_text = game.n_players_text
    
    
    fixlip = src.fixlip.FIxLIP(
    n_players_image=n_players_image,
    n_players_text=n_players_text, 
    max_order=2,
    p=0.5, # weight
    mode="banzhaf",
    random_state=0,
    approximation_type="regression"
    )
    fixlip_proxyshap = src.fixlip.FIxLIP(
        n_players_image=n_players_image,
        n_players_text=n_players_text,
        max_order=2,
        p=0.5,  # weight
        mode="banzhaf",
        random_state=0,
        approximation_type="proxyshap",
    )
    flixlip_proxyspex = src.fixlip.FIxLIP(
        n_players_image=n_players_image,
        n_players_text=n_players_text,
        max_order=2,
        p=0.5,  # weight
        mode="banzhaf",
        random_state=0,
        approximation_type="proxyspex",
    )
    flixlip_surrogate = src.fixlip.FIxLIP(
        n_players_image=n_players_image,
        n_players_text=n_players_text,
        max_order=2,
        p=0.5,  # weight
        mode="banzhaf",
        random_state=0,
        approximation_type="proxyshap-noadjustment",
    )
    
    import time


    src.utils.set_seed(0)
    BUDGET = 10_000
    a = time.time()
    interaction_values = fixlip.approximate_crossmodal(game, budget=BUDGET)
    b = time.time()
    print("FIxLIP runtime (s): ", b - a)
    interaction_values_proxyshap = fixlip_proxyshap.approximate_crossmodal(game, budget=BUDGET)
    c = time.time()
    print("FIxLIP ProxySHAP runtime (s): ", c - b)
    interaction_values_proxyspex = flixlip_proxyspex.approximate_crossmodal(game, budget=BUDGET)
    d = time.time()
    print("FIxLIP ProxySpex runtime (s): ", d - c)
    interaction_values_surrogate = flixlip_surrogate.approximate_crossmodal(game, budget=BUDGET)
    e = time.time()
    print("FIxLIP Surrogate runtime (s): ", e - d)
    
    
    from src.evaluation import eval_faithfulness_one_game
    results = eval_faithfulness_one_game(
        game=game,
        explanations={
            "fixlip": interaction_values,
            "fixlip_proxyshap": interaction_values_proxyshap,
            "fixlip_proxyspex": interaction_values_proxyspex,
            "flxlip_surrogate": interaction_values_surrogate
        },
        n_eval_coalitions=1000,
        sample_mode="banzhaf",
        sample_p=0.5,
    )
    
    for res in results:
        print("Results for Method:", res["method_name"], "with budget = ", BUDGET)
        for metric in ["r2", "r2_banzhaf", "mse","mae","correlation", "cosine_similarity"]:
            print(f"  {metric}: {res[metric]:.4f}")
