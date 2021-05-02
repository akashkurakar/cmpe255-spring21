from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


class SVM:

    def load_data(self):
        self.faces = fetch_lfw_people(min_faces_per_person=60)
        return self.faces

    def create_model(self):
        pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
        svc = SVC(kernel='rbf', class_weight='balanced')
        model = make_pipeline(pca, svc)
        return model

    def img_data(self):
        n_samples, h, w = self.faces.images.shape
        X = self.faces.data
        n_features = X.shape[1]
        y = self.faces.target
        target_names = self.faces.target_names
        n_classes = target_names.shape[0]
        print("Total dataset size:")
        print("n_samples: %d" % n_samples)
        print("n_features: %d" % n_features)
        print("n_classes: %d" % n_classes)
        return h, w

    def train(self):
        X = self.faces.data
        y = self.faces.target
        model = self.create_model()
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=0)

        C_range = np.logspace(-1, 5, 4)
        gamma_range = np.logspace(-3, 0, 4)
        grid_values = dict(svc__gamma=gamma_range, svc__C=C_range)
        clf = GridSearchCV(
            model, grid_values
        )
        clf.fit(X_train, y_train)
        self.y_pred = clf.predict(self.X_test)
        print(classification_report(self.y_test, self.y_pred,
                                    target_names=self.faces.target_names))

    def plot_confusion_matrix(self):
        cmap = plt.cm.Blues
        cm = confusion_matrix(self.faces.target_names[self.y_pred],
                              self.faces.target_names[self.y_test])

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(self.faces.target_names))
        plt.xticks(tick_marks, self.faces.target_names, rotation=45)
        plt.yticks(tick_marks, self.faces.target_names)

        fmt = ".2f"
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig('heatmap')

    def plot_suite(self, h, w):
        images = self.X_test
        titles = self.faces.target_names[self.y_pred]
        names_actual = self.faces.target_names[self.y_test]
        n_row = 4
        n_col = 6
        fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            ax = fig.add_subplot(n_row, n_col, i + 1)
            ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            fontColor = 'black'
            if titles[i] != names_actual[i]:
                fontColor = 'red'
            title = "Predicted: "+titles[i]+"\nActual: "+names_actual[i]
            ax.set_title(titles[i], size=12, color=fontColor)
            plt.xticks(())
            plt.yticks(())

        fig.suptitle("Predictions"+'\n', fontsize=20)
        plt.savefig('Gallery')


def test() -> None:
    svm = SVM()
    svm.load_data()
    svm.train()
    h, w = svm.img_data()
    svm.plot_confusion_matrix()
    svm.plot_suite(h, w)


if __name__ == "__main__":
    # execute only if run as a script
    test()
