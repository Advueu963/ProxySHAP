from __future__ import annotations
import argparse
from pathlib import Path
import os
import numpy as np
from shapiq_games.benchmark import (AdultCensusLocalXAI,
                                    BikeSharingLocalXAI, 
                                    ForestFiresLocalXAI, 
                                    RealEstateLocalXAI, 
                                    CaliforniaHousingLocalXAI
                                    )



if __name__ == "__main__":
    """This scipt prints the accuracy of the TabPFN Model used for the TabPFN Explanation games-"""
    RANDOM_STATE = 40  # random state for the games
    games = [
        AdultCensusLocalXAI,
        BikeSharingLocalXAI,
        ForestFiresLocalXAI,
        RealEstateLocalXAI,
        CaliforniaHousingLocalXAI,
    ]
    results = {}
    for game_class in games:
        game = game_class(
            model_name="tabpfn",
            imputer="tabpfn",
            random_state=RANDOM_STATE,
        )
        
        model = game.setup.model
        x_test = game.setup.x_test
        y_test = game.setup.y_test
        problem_type = game.setup.dataset_type
        if problem_type == "classification":
            # Print Accuracy
            print(f"TabPFN Model Accuracy on {game_class.__name__} Test Set: ")
            print(np.mean(model.predict(x_test) == y_test))
            print("-----")
            results[game_class.__name__] = np.mean(model.predict(x_test) == y_test)
        else:
            # Print R^2 Score
            from sklearn.metrics import r2_score
            print(f"TabPFN Model R^2 Score on {game_class.__name__} Test Set: ")
            print(r2_score(y_test, model.predict(x_test)))
            print("-----")
            results[game_class.__name__] = r2_score(y_test, model.predict(x_test))
        
    # Save results
    import pandas as pd
    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["TabPFN_performance"])
    results_df.to_csv("tabpfn_performance_results.csv")
    # Print latex table
    print(results_df.to_latex())