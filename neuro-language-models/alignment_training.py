import argparse
import os

import utils

import numpy as np
import pytorch_lightning as L

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from loguru import logger
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--subj', type=int, default=0)
    # parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()
    
    # read config file
    config = utils.read_yaml(args.config)
    L.seed_everything(config['SEED'])
    
    X_train, Y_train, X_test, Y_test = utils.load_data(subj=args.subj, config=config)
    logger.info(f'Memory size is {config["DATA"]["memory_size"]}')
    if config['ALIGNMENT_TRAINING']['model']=='ridge_regression':
        assert config['DATA']['type']=='array', "Only array data type is supported for ridge regression"
        
        config['ALIGNMENT_TRAINING']['save_path'] = f"{config['ALIGNMENT_TRAINING']['save_path']}_mem_{config['DATA']['memory_size']}"
        os.makedirs(config['ALIGNMENT_TRAINING']['save_path'], exist_ok=True)
              
        mse_loss = 0.0
        corr_score = 0.0
        for chn in range(32): # num blip channels
            if os.path.exists(os.path.join(config['ALIGNMENT_TRAINING']['save_path'], 
                                                    f'model_{chn}.pkl')):
                logger.warning(f"Model for channel {chn} already exists. Skipping...")
                continue
            
            start = time.time()
            x_train, y_train, x_test, y_test = utils.load_data_as_array(
                X_train, Y_train, X_test, Y_test, subj=args.subj, config=config, channel=chn
            )
            
            best_model = None
            best_corr = -1.0
            best_mse = 1e10
            Y_test_pred_best = None
            best_alpha = None
            
            for alpha in config['ALIGNMENT_TRAINING']['alphas']:
                model = Ridge(alpha=alpha,
                                fit_intercept=True,
                                max_iter=50000)
                model.fit(x_train, y_train)
                Y_test_pred = model.predict(x_test)
                
                test_corr = np.corrcoef(Y_test_pred, y_test)[0, 1]
                test_mse = mean_squared_error(Y_test_pred, y_test)
                
                if test_mse < best_mse:
                    best_mse = test_mse
                    best_corr = test_corr
                    best_model = model
                    Y_test_pred_best = Y_test_pred
                    best_alpha = alpha
            
            mse_loss += best_mse
            corr_score += best_corr
            
            minutes, seconds = divmod(time.time()-start, 60)
            logger.info(f"For channel: {chn} |  Test corr: {best_corr:.4f} | Test mse: {best_mse:.4f}")
            logger.info(f"Best alpha: {best_alpha}")
            logger.info(f"Time taken: {minutes:.0f}m {seconds:.0f}s")
            
            utils.save_pickle(best_model, os.path.join(config['ALIGNMENT_TRAINING']['save_path'], 
                                                    f'model_{chn}.pkl'))
            utils.save_pickle(Y_test_pred_best, os.path.join(config['ALIGNMENT_TRAINING']['save_path'], 
                                                    f'Y_test_pred_{chn}.pkl'))
            logger.success(f"Models for channel {chn} saved to {config['ALIGNMENT_TRAINING']['save_path']}")
            
        mse_loss /= 32
        corr_score /= 32
        logger.info(f"Mean squared error loss: {mse_loss:.4f}")
        logger.info(f"Mean correlation score: {corr_score:.4f}")
               
    