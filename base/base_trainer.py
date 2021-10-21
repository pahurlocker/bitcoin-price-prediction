from plotly.offline import plot, iplot
import plotly.graph_objs as go
import plotly.express as px

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    classification_report,
    auc,
)


class BaseTrain(object):
    def __init__(self, model, train_data, test_data, config):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.config = config

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def visualize_prediction(self, y_test, X_test_pred, metric=0):

        if self.config.trainer.model_task == "regression":
            trace1 = go.Scatter(
                x=np.arange(len(y_test)), y=y_test, mode="lines", name="Original Price"
            )
            trace2 = go.Scatter(
                x=np.arange(len(X_test_pred)),
                y=X_test_pred,
                mode="lines",
                name="Predicted Price",
            )
            layout = dict(
                title="Original vs Predicted Values Curve, MAPE: " + str(metric),
                xaxis=dict(rangeslider=dict(visible=True), type="linear"),
                yaxis=dict(title="BTC Price USD"),
            )
            pdata = [trace1, trace2]
            fig = dict(data=pdata, layout=layout)
            iplot(fig, filename="Time Series with Rangeslider")
        else:
            # Performance Metrics
            X_pred_rf = self.model.predict_proba(y_test, self.config.trainer.batch_size)
            fpr_rf, tpr_rf, _ = roc_curve(y_test, X_pred_rf)
            print(classification_report(y_test, X_test_pred))

            print("")
            print("Accuracy")
            print(accuracy_score(y_test, X_test_pred))

            fig = px.area(
                x=fpr_rf,
                y=tpr_rf,
                title=f"ROC Curve (AUC={auc(fpr_rf, tpr_rf):.4f})",
                labels=dict(x="False Positive Rate", y="True Positive Rate"),
                width=700,
                height=500,
            )
            fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_xaxes(constrain="domain")
            iplot(fig, filename="ROC Curve")
