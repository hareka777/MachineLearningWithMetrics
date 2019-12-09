using App.Metrics;
using MachineLearningWithMetrics.Metrics;
using MachineLearningWithMetrics.MLdotNET.DataModel.Eurorate;
using Microsoft.ML;
using System.Windows;
using static MachineLearningWithMetrics.MLdotNET.Predictors.Common_Classes.Algorithms;
using static Microsoft.ML.DataOperationsCatalog;

namespace MachineLearningWithMetrics.MLdotNET.Predictors
{
    public class EuroPredictor : IPredictor
    {
        #region Fields and Properties
        private static readonly string dataFolderPath = Paths.dataFolderPath + @"\Euro";
        private readonly string dataModelFolderPath = Paths.dataModelFolderPath;
        private readonly string networkPath = Paths.networkModelFolderPath + @"\Euro.zip";
        private IDataView trainingData;
        private IDataView testData;
        private ITransformer trainedModel;

        private string dataPath = dataFolderPath + @"\Euro24hrData.csv";

        public RegressionTrainingAlgorithm TrainingAlgo
        {
            get;
            set;
        }

        IMetricsRoot _metrics;

        #endregion

        #region Constructor

        public EuroPredictor()
        {
            _metrics = MetricsInitializer.Metrics;
            trainTestDataRate = 0.2;
            TrainingAlgo = RegressionTrainingAlgorithm.OnlineGradientDescent;          
        }
        #endregion

        #region Private Methods
        public override void ProcessNetwork()
        {
            MLContext mlContext = new MLContext();

            IDataView dataView = LoadData(mlContext, dataPath);

            TrainTestData allData = mlContext.Data.TrainTestSplit(dataView, testFraction: trainTestDataRate);
            trainingData = allData.TrainSet;
            testData = allData.TestSet;

            var dataprocessPipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(EuroDataModel.Value))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev1)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev2)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev3)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev4)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev5)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev6)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev7)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev8)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev9)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev10)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev11)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev12)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev13)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev14)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev15)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev16)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev17)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev18)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev19)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev20)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev21)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev22)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev23)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(EuroDataModel.DiffPrev24)))
                            .Append(mlContext.Transforms.Concatenate("Features", nameof(EuroDataModel.DiffPrev1), nameof(EuroDataModel.DiffPrev2), nameof(EuroDataModel.DiffPrev3), nameof(EuroDataModel.DiffPrev4), nameof(EuroDataModel.DiffPrev5),
                nameof(EuroDataModel.DiffPrev6), nameof(EuroDataModel.DiffPrev7), nameof(EuroDataModel.DiffPrev8), nameof(EuroDataModel.DiffPrev9), nameof(EuroDataModel.DiffPrev10), nameof(EuroDataModel.DiffPrev11), nameof(EuroDataModel.DiffPrev12), 
                nameof(EuroDataModel.DiffPrev13), nameof(EuroDataModel.DiffPrev14), nameof(EuroDataModel.DiffPrev15), nameof(EuroDataModel.DiffPrev16), nameof(EuroDataModel.DiffPrev17), nameof(EuroDataModel.DiffPrev18), nameof(EuroDataModel.DiffPrev19),
                nameof(EuroDataModel.DiffPrev20), nameof(EuroDataModel.DiffPrev21), nameof(EuroDataModel.DiffPrev22), nameof(EuroDataModel.DiffPrev23), nameof(EuroDataModel.DiffPrev24)));

            //Applying training algorithm
            var trainer = ApplyRegressionTrainingAlgorithm(mlContext, TrainingAlgo);
            var trainingPipeline = dataprocessPipeline.Append(trainer);

            //Training the model
            string[] tags = new string[]{
                this.ToString(),
                this.TrainingAlgo.ToString(),
                "Training the model"
            };
            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, MetricsTags.CreateMetricsTags(tags)))
            {
                trainedModel = trainingPipeline.Fit(trainingData);
            }



            EvaluateModel(trainedModel);

            SaveNetwork(trainedModel);

            TestSomePredictions();      
        }
        #endregion

        internal override IDataView LoadData(MLContext context, string dataPath)
        {

            IDataView loadedData = null;
            //loading TrainingData
            string[] tags = new string[]{
                this.ToString(),
                this.TrainingAlgo.ToString(),
                "Loading Data"
            };

            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, MetricsTags.CreateMetricsTags(tags)))
            {
                loadedData = context.Data.LoadFromTextFile<EuroDataModel>(path: dataPath,
                       hasHeader: true,
                       separatorChar: ','
                       );
            }

            return loadedData;

        }

        public override string ToString()
        {
            return "Euro price(Regression)";
        }

        internal override void EvaluateModel(ITransformer trainedModel)
        {
            string[] tags = new string[]{
                this.ToString(),
                this.TrainingAlgo.ToString(),
                "Evaluating Network"
            };
            IDataView predictions;
            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, MetricsTags.CreateMetricsTags(tags)))
            {
                predictions = trainedModel.Transform(testData);
            }
            var metrics = mlContext.Regression.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

            _metrics.Measure.Gauge.SetValue(MetricsRegistry.TrainTestRate, MetricsTags.CreateMetricsTags(new string[] { "Network" }, new string[] { nameof(EuroPredictor) }), this.TrainTestDataRate);
            _metrics.Measure.Gauge.SetValue(MetricsRegistry.NetworkEvaluatingResult, MetricsTags.CreateMetricsTags(new string[] { "Network", "Algorithm", "Metric", "TrainTestRate" }, new string[] { nameof(EuroPredictor), this.TrainingAlgo.ToString(), "MeanAbsoluteError", this.TrainTestDataRate.ToString() }), metrics.MeanAbsoluteError);

        }

        internal override void SaveNetwork(ITransformer trainedModel)
        {
            string[] tags = new string[]{
                this.ToString(),
                this.TrainingAlgo.ToString(),
                "Saving Network"
            };
            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, MetricsTags.CreateMetricsTags(tags)))
            {
                mlContext.Model.Save(trainedModel, trainingData.Schema, networkPath);
            }
        }

        internal override void TestSomePredictions()
        {
            var loadedtrainedModel = mlContext.Model.Load(networkPath, out var modelInputSchema);
            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<EuroDataModel, EuroDataModelOutput>(loadedtrainedModel);
            //
            var resultprediction1 = predEngine.Predict(SampleEuroData.Euro1);
            MessageBox.Show($"Value was: 310.085, the predicted value is:{resultprediction1.Score}");

            var resultprediction2 = predEngine.Predict(SampleEuroData.Euro2);
            MessageBox.Show($"Value was: 323.923, the predicted value is:{resultprediction2.Score}");
        }

        public override void SetAlgorithm(object algo)
        {
            this.TrainingAlgo = (RegressionTrainingAlgorithm)algo;
        }
    }
}

