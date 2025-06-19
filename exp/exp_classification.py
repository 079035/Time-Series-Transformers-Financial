from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from utils.losses import FocalLoss, WeightedBCEWithLogitsLoss, WeightedCrossEntropyLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """Select the appropriate loss function based on args.class_loss"""
        if self.args.class_loss == 'CE':
            criterion = nn.CrossEntropyLoss()
        elif self.args.class_loss == 'focal':
            # For binary classification with severe imbalance, use alpha to weight classes
            if self.args.num_class == 2:
                # alpha weights for [negative, positive] classes
                alpha = [1 - self.args.focal_alpha, self.args.focal_alpha]
            else:
                alpha = None
            criterion = FocalLoss(alpha=alpha, gamma=self.args.focal_gamma)
        elif self.args.class_loss == 'weighted_ce':
            # Parse class weights from string
            weights = [float(w) for w in self.args.class_weights.split(',')]
            if len(weights) != self.args.num_class:
                raise ValueError(f"Number of class weights ({len(weights)}) must match number of classes ({self.args.num_class})")
            weight_tensor = torch.tensor(weights, device=self.device)
            criterion = WeightedCrossEntropyLoss(weight=weight_tensor)
        elif self.args.class_loss == 'weighted_bce':
            if self.args.num_class != 2:
                raise ValueError("Weighted BCE loss only supports binary classification")
            criterion = WeightedBCEWithLogitsLoss(pos_weight=self.args.pos_weight)
        else:
            raise ValueError(f"Unknown classification loss: {self.args.class_loss}")
        
        print(f"Using {self.args.class_loss} loss function")
        if self.args.class_loss == 'focal':
            print(f"  Focal loss parameters: alpha={self.args.focal_alpha}, gamma={self.args.focal_gamma}")
        elif self.args.class_loss == 'weighted_bce':
            print(f"  Weighted BCE pos_weight={self.args.pos_weight}")
        elif self.args.class_loss == 'weighted_ce':
            print(f"  Class weights: {self.args.class_weights}")
            
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                # Compute loss on the same device as the model/criterion
                loss = criterion(outputs, label.long().squeeze(-1))
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        
        # Calculate PR AUC for binary classification
        pr_auc = 0.0
        if self.args.num_class == 2:
            pos_probs = probs[:, 1].cpu().numpy()
            pr_auc = average_precision_score(trues, pos_probs)

        self.model.train()
        return total_loss, accuracy, pr_auc

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        
        # Try to get validation data, but make it optional
        try:
            vali_data, vali_loader = self._get_data(flag='VAL')
            has_validation = True
        except:
            print("No validation data found, training without validation")
            vali_data, vali_loader = None, None
            has_validation = False
            
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) if has_validation else None

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            # Compute metrics for all sets including training
            train_eval_loss, train_accuracy, train_pr_auc = self.vali(train_data, train_loader, criterion)
            test_loss, test_accuracy, test_pr_auc = self.vali(test_data, test_loader, criterion)

            if has_validation:
                vali_loss, val_accuracy, val_pr_auc = self.vali(vali_data, vali_loader, criterion)
                
                if self.args.num_class == 2:
                    print(
                        "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Train Acc: {3:.3f} Train PR-AUC: {4:.3f} | Vali Loss: {5:.3f} Vali Acc: {6:.3f} Vali PR-AUC: {7:.3f} | Test Loss: {8:.3f} Test Acc: {9:.3f} Test PR-AUC: {10:.3f}"
                        .format(epoch + 1, train_steps, train_loss, train_accuracy, train_pr_auc, vali_loss, val_accuracy, val_pr_auc, test_loss, test_accuracy, test_pr_auc))
                    # For binary classification, use PR-AUC for early stopping instead of accuracy
                    early_stopping(-val_pr_auc, self.model, path)
                else:
                    print(
                        "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Train Acc: {3:.3f} | Vali Loss: {4:.3f} Vali Acc: {5:.3f} | Test Loss: {6:.3f} Test Acc: {7:.3f}"
                        .format(epoch + 1, train_steps, train_loss, train_accuracy, vali_loss, val_accuracy, test_loss, test_accuracy))
                    early_stopping(-val_accuracy, self.model, path)
                    
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            else:
                # No validation set - just print train and test metrics
                if self.args.num_class == 2:
                    print(
                        "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Train Acc: {3:.3f} Train PR-AUC: {4:.3f} | Test Loss: {5:.3f} Test Acc: {6:.3f} Test PR-AUC: {7:.3f}"
                        .format(epoch + 1, train_steps, train_loss, train_accuracy, train_pr_auc, test_loss, test_accuracy, test_pr_auc))
                else:
                    print(
                        "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Train Acc: {3:.3f} | Test Loss: {4:.3f} Test Acc: {5:.3f}"
                        .format(epoch + 1, train_steps, train_loss, train_accuracy, test_loss, test_accuracy))
                
                # Save model after each epoch when no validation (since no early stopping)
                torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # Final evaluation and logging
        final_train_loss, final_train_accuracy, final_train_pr_auc = self.vali(train_data, train_loader, criterion)
        
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('Train Final Results:\n')
        f.write('Train Accuracy: {:.4f}\n'.format(final_train_accuracy))
        if self.args.num_class == 2:
            f.write('Train PR AUC: {:.4f}\n'.format(final_train_pr_auc))
        
        if has_validation:
            final_vali_loss, final_val_accuracy, final_val_pr_auc = self.vali(vali_data, vali_loader, criterion)
            f.write('Validation Final Results:\n')
            f.write('Validation Accuracy: {:.4f}\n'.format(final_val_accuracy))
            if self.args.num_class == 2:
                f.write('Validation PR AUC: {:.4f}\n'.format(final_val_pr_auc))
        f.close()

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        
        # Calculate PR AUC for binary classification
        pr_auc = None
        avg_precision = None
        if self.args.num_class == 2:  # Binary classification
            # Get probabilities for positive class (class 1)
            pos_probs = probs[:, 1].cpu().numpy()
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(trues, pos_probs)
            pr_auc = auc(recall, precision)
            
            # Also calculate average precision score
            avg_precision = average_precision_score(trues, pos_probs)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('Accuracy: {:.4f}'.format(accuracy))
        if pr_auc is not None:
            print('PR AUC: {:.4f}'.format(pr_auc))
            print('Average Precision: {:.4f}'.format(avg_precision))
            
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('Test Final Results:\n')
        f.write('Accuracy: {:.4f}\n'.format(accuracy))
        if pr_auc is not None:
            f.write('PR AUC: {:.4f}\n'.format(pr_auc))
            f.write('Average Precision: {:.4f}\n'.format(avg_precision))
        f.write('\n')
        f.close()
        return
