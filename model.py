from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import HistGradientBoostingClassifier

import torch
from torchinfo import summary
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neuralnet import NeuralNet
from dataset import CustomDataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader

from imblearn.over_sampling import SMOTE

class Model():
    def __init__(self, args):
        ############################ Your Code Here ############################
        # Initialize your model in this space
        # You can add arguements to the initialization as needed
        self.scaler = None
        self.normalize_features = None
        self.columns = None

        self.encoded_columns = ['ethnicity', 'cellattributevalue']
        self.normalized_columns = ['age', 'admissionheight', 'admissionweight', 'bmi', 'offset', 'unitvisitnumber']
        self.pivot_columns = ['nursingchartvalue', 'labresult']

        self.model = None
        self.batch_size = 128
        self.lr = 0.00004
        ########################################################################

    def model_creation(self):
        self.model = NeuralNet()
        summary(self.model, input_size=(self.batch_size, 1, 30))

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        ############################ Your Code Here ############################

        # prepping data
        self.columns = x_train.columns

        x_train = self.train_normalize(x_train)
        x_train = self.aggregate(x_train)

        x_train = x_train.sort_values('patientunitstayid')
        y_train = y_train.sort_values('patientunitstayid')

        x_train = x_train.drop('patientunitstayid', axis=1)
        y_train = y_train.drop('patientunitstayid', axis=1)

        # define model

        self.model_creation()

        # train model

        import torch.optim as optim

        def binary_cross_entropy_loss(inputs, targets):
            # weight = torch.tensor([1, 9], dtype=torch.float).to(device)
            # return F.binary_cross_entropy_with_logits(inputs, targets, weight=weight)
            return F.binary_cross_entropy(inputs, targets)

        num_epochs = 150
        loss_fn = binary_cross_entropy_loss
        opt_fn = optim.Adam(self.model.parameters(), self.lr)

        dataset = self.to_dataset(x_train, y_train)

        val_frac =  0.2
        rand_seed =  42
        train_indices, val_indices = self.split_indices(len(dataset), val_frac, rand_seed)


        # targets = dataset.targets[train_indices]
        # targets = targets.squeeze()
        # class_counts = torch.bincount(targets.long())
        # class_weights = 1.0 / class_counts.float()
        # class_weights /= torch.sum(class_weights)
        # weights = class_weights[targets.long()]
        # train_sampler = WeightedRandomSampler(weights, len(weights))

        train_sampler = SubsetRandomSampler(train_indices)
        train_dl = DataLoader(dataset,
                            self.batch_size,
                            sampler=train_sampler,
                            drop_last=True)

        val_sampler = SubsetRandomSampler(val_indices)
        val_dl = DataLoader(dataset,
                        self.batch_size,
                        sampler=val_sampler,
                        drop_last=True)

        def get_default_device():
            """Use GPU if available, else CPU"""
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')

        def to_device(data, device):
            """Move tensor(s) to chosen device"""
            if isinstance(data, (list,tuple)):
                return [to_device(x, device) for x in data]
            return data.to(device, non_blocking=True)

        class DeviceDataLoader():
            """Wrap a dataloader to move data to a device"""
            def __init__(self, dl, device):
                self.dl = dl
                self.device = device

            def __iter__(self):
                """Yield a batch of data after moving it to device"""
                for b in self.dl:
                    yield to_device(b, self.device)

            def __len__(self):
                """Number of batches"""
                return len(self.dl)

        device = get_default_device()

        train_dl = DeviceDataLoader(train_dl, device)
        val_dl = DeviceDataLoader(val_dl, device)

        to_device(self.model, device)

        self.train_model(num_epochs, train_dl, val_dl, loss_fn, opt_fn)

        self.model = NeuralNet()
        to_device(self.model, device)

        indices, _ = self.split_indices(len(dataset), 0, rand_seed)
        sampler = SubsetRandomSampler(indices)
        dl = DataLoader(dataset, self.batch_size, sampler=sampler, drop_last=True)
        dl = DeviceDataLoader(dl, device)

        opt_fn = optim.Adam(self.model.parameters(), self.lr)
        self.train_model(num_epochs, dl, [], loss_fn, opt_fn)
        ########################################################################

    def predict_proba(self, x):
        ############################ Your Code Here ############################
        # Predict the probability of in-hospital mortaility for each x

        x = self.fix_columns(self.columns, x)
        x = self.test_normalize(x)
        x = self.aggregate(x)

        ids = x['patientunitstayid']
        x = x.drop('patientunitstayid', axis=1)

        x = x.fillna(0)
        x = x.astype(np.float32).to_numpy()
        x = torch.from_numpy(x)

        preds = []
        with torch.no_grad():
            self.model.eval()
            preds = self.model(x)

        df_combined = pd.concat([ids, pd.Series(preds.flatten().tolist())], axis=1)

        cols = df_combined.columns.tolist()
        cols[1] = 'hospitaldischargestatus'
        df_combined.columns = cols

        df_combined.to_csv("submission.csv", index=False, float_format='%.14f')

        ########################################################################
        return preds

    # def score(self, x_val, y_val):
    #     x_val = self.fix_columns(self.columns, x_val)
    #     x_val = self.test_normalize(x_val)
    #     x_val = self.aggregate(x_val)

    #     x_val = x_val.sort_values('patientunitstayid')
    #     y_val = y_val.sort_values('patientunitstayid')

    #     x_val.to_csv("x_val.csv", index=False)

    #     x_val = x_val.drop('patientunitstayid', axis=1)

    #     pred_proba = self.model.predict_proba(x_val)

    #     score = roc_auc_score(y_val['hospitaldischargestatus'], pred_proba[:,1])
    #     print('ROC-AOC:', score)

    def train_normalize(self, df):
        # pivot creates new columns, these columns need to be normalized too
        for c in df.columns:
            if "nursingchartvalue" in c or "labresult" in c:
                self.normalized_columns.append(c)

        # normalizing
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        normalized_data = self.scaler.fit_transform(df[self.normalized_columns])
        normalized_df = pd.DataFrame(normalized_data, columns=self.normalized_columns)
        df[self.normalized_columns] = normalized_df

        return df

    def test_normalize(self, df):
        # normalizing
        normalized_data = self.scaler.transform(df[self.normalized_columns])
        normalized_df = pd.DataFrame(normalized_data, columns=self.normalized_columns)
        df[self.normalized_columns] = normalized_df

        return df

    def fix_columns(self, train_x_columns, test_df):
        if not set(train_x_columns) == set(test_df.columns):
            missing_columns = set(train_x_columns) - set(test_df.columns)

            for column in missing_columns:
                # if self.is_column_encoded(column):
                #     test_df[column] = 0
                # else:
                #     test_df[column] = np.nan

                test_df[column] = np.nan

            additional_columns = set(test_df.columns) - set(train_x_columns)
            test_df = test_df.drop(columns=additional_columns)
            test_df = test_df.sort_index(axis=1)

        return test_df

    def is_column_encoded(self, column_name):
        for c in self.encoded_columns:
            if column_name in c:
                return True
        return False

    def aggregate(self, df):

        # def furthest_from_zero(x):
        #     return x.iloc[np.abs(x).argmax()]

        df = df.groupby('patientunitstayid', as_index=False).agg(np.nanmean)
        df = df.reset_index(drop=True)
        return df

    def train_model(self, n_epochs, train_dl, val_dl, loss_fn, opt_fn):
        """
        Trains the model on a dataset.

        Args:
            n_epochs: number of epochs
            model: NeuralNet object
            train_dl: training dataloader
            val_dl: validation dataloader
            loss_fn: the loss function
            opt_fn: the optimizer
            lr: learning rate

        Returns:
            The trained model.
            A tuple of (model, train_losses, val_losses, train_accuracies, val_accuracies)
        """
        train_losses, val_losses, train_accuracies, val_accuracies, train_roc_aucs, val_roc_aucs = [], [], [], [], [], []

        for epoch in range(n_epochs):
            train_loss, count_correct, count_total = 0, 0, 0
            y_true_train, y_pred_train = [], []

            self.model.train()

            for x, y in train_dl:
                opt_fn.zero_grad()

                y_pred_prob = self.model(x)
                y_pred = (y_pred_prob>= 0.5).float()
                loss = loss_fn(y_pred_prob, y.float())
                # y_pred = (y_pred_prob[:, 1] >= 0.5).float()
                # targets = torch.cat([1-y, y], dim=1)
                # loss = loss_fn(y_pred_prob, targets.float())


                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                opt_fn.step()

                train_loss += len(x) * loss.item()
                count_correct += (y_pred == y.float()).sum().item()
                count_total += len(x)

                y_true_train += y.cpu().numpy().tolist()
                # y_pred_train += y_pred_prob[:, 1].detach().numpy().tolist()
                y_pred_train += y_pred_prob.detach().numpy().tolist()

            train_loss = train_loss / count_total
            train_accuracy = count_correct / count_total
            train_roc_auc = roc_auc_score(y_true_train, y_pred_train)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_roc_aucs.append(train_roc_auc)

            if len(val_dl) > 0:
                val_loss, count_correct, count_total = 0, 0, 0
                y_true_val, y_pred_val = [], []

                self.model.eval()

                with torch.no_grad():
                    for x, y_true in val_dl:
                        y_pred_prob = self.model(x)
                        y_pred = (y_pred_prob >= 0.5).float()
                        loss = loss_fn(y_pred_prob, y.float())
                        # y_pred = (y_pred_prob[:, 1] >= 0.5).float()
                        # targets = torch.cat([1-y, y], dim=1)
                        # loss = loss_fn(y_pred_prob, targets.float())

                        val_loss += loss.item() * len(x)
                        count_correct += (y_pred == y_true.float()).sum().item()
                        count_total += len(x)

                        y_true_val += y_true.cpu().numpy().tolist()
                        y_pred_val += y_pred_prob.cpu().numpy().tolist()
                        # y_pred_val += y_pred_prob[:, 1].cpu().numpy().tolist()

                val_loss = val_loss / count_total
                val_accuracy = count_correct / count_total
                val_roc_auc = roc_auc_score(y_true_val, y_pred_val)

                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                val_roc_aucs.append(val_roc_auc)

            # Print progress
            if len(val_dl) > 0:
                print("Epoch {}/{}, train_loss: {:.4f}, val_loss: {:.4f}, train_accuracy: {:.4f}, val_accuracy: {:.4f}, train_roc_auc: {:.4f}, val_roc_auc: {:.4f}"
                        .format(epoch+1, n_epochs, train_loss, val_loss, train_accuracy, val_accuracy, train_roc_auc, val_roc_auc))
            else:
                print("Epoch {}/{}, train_loss: {:.4f}, train_accuracy: {:.4f}, train_roc_auc: {:.4f}"
                        .format(epoch+1, n_epochs, train_loss, train_accuracy, train_roc_auc))

        if len(val_dl) > 0:
            plt.plot(train_roc_aucs, "-x")
            plt.plot(val_roc_aucs, "-o")
            plt.xlabel("Epoch")
            plt.ylabel("ROC AUC")
            plt.legend(["Training", "Validation"])
            plt.title("ROC AUC vs. No. of Epochs")
            plt.show()

            plt.plot(train_losses, "-x")
            plt.plot(val_losses, "-o")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(["Training", "Validation"])
            plt.title("Loss vs. No. of Epochs")
            plt.show()

    def split_indices(self, n, val_frac, seed):
        # Determine the size of the validation set
        n_val = int(val_frac * n)
        np.random.seed(seed)
        # Create random permutation between 0 to n-1
        idxs = np.random.permutation(n)
        # Pick first n_val indices for validation set
        return idxs[n_val:], idxs[:n_val]

    def to_dataset(self, x, y):

        x = x.fillna(0)

        x = x.astype(np.float32).to_numpy()
        y = y.astype(np.int64).to_numpy()

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        dataset = CustomDataset(x, y)
        return dataset
