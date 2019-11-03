using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace MachineLearningWithMetrics.MLdotNET.Predictors
{
    public abstract class IPredictor
    {
        #region Fields
        internal IDataView trainingData = null;
        internal IDataView testData = null;
        internal ITransformer trainedModel = null;
        internal MLContext mlContext = new MLContext();
        #endregion

        #region Internal Methods
        internal abstract IDataView LoadData(MLContext context, string dataPath);
        internal abstract void EvaluateModel(ITransformer trainedModel);
        internal abstract void SaveNetwork(ITransformer trainedModel);
        internal abstract void ProcessNetwork();
        internal abstract void TestSomePredictions();
        internal IDataView ShuffleData(IDataView loadedData)
        {
            return mlContext.Data.ShuffleRows(loadedData, seed: 37);
        }
        #endregion
    }
}
