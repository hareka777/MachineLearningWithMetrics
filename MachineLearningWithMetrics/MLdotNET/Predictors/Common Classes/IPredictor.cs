using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;

namespace MachineLearningWithMetrics.MLdotNET.Predictors
{
    public abstract class IPredictor
    {
        #region Fields
        internal IDataView trainingData = null;
        internal IDataView testData = null;
        internal ITransformer trainedModel = null;
        internal MLContext mlContext = new MLContext();

        internal double trainTestDataRate;

        public double TrainTestDataRate
        {
            get { return trainTestDataRate; }
            set { trainTestDataRate = value; }
        }
        #endregion

        #region Methods
        internal abstract IDataView LoadData(MLContext context, string dataPath);
        internal abstract void EvaluateModel(ITransformer trainedModel);
        internal abstract void SaveNetwork(ITransformer trainedModel);
        internal abstract void TestSomePredictions();
        public abstract void ProcessNetwork();
        public abstract void SetAlgorithm(object algo);

        internal void AppendTrainingTestDataRate(IDataView data)
        {
            TrainTestData allData = mlContext.Data.TrainTestSplit(data, testFraction: trainTestDataRate);
            trainingData = allData.TrainSet;
            testData = allData.TestSet;
        }
        #endregion
    }
}
