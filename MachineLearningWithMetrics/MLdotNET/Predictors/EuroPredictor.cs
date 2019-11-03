using App.Metrics;
using MachineLearningWithMetrics.Metrics;
using MachineLearningWithMetrics.MLdotNET.DataModel.Eurorate;
using MachineLearningWithMetrics.MLdotNET.DataModel.MNIST;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Windows;

namespace MachineLearningWithMetrics.MLdotNET.Predictors
{
    public class EuroPredictor
    {
        #region Fields
        private static readonly string dataFolderPath = Paths.dataFolderPath + @"\Euro";
        private readonly string dataModelFolderPath = Paths.dataModelFolderPath;
        private readonly string networkPath = Paths.networkModelFolderPath + @"\Euro.zip";


        private string trainingDataPath = dataFolderPath + @"\EuroRateTrainingShort.csv";
        private string testDataPath = dataFolderPath + @"\EuroRateTest.csv";

        IMetricsRoot _metrics;
        #endregion

        #region Constructor

        public EuroPredictor()
        {
            _metrics = MetricsInitializer.Metrics;
            ProcessNetwork();
        }
        #endregion

        #region Private Methods
        private void ProcessNetwork()
        {
            MLContext mlContext = new MLContext();
            var trainingDataView = LoadData(mlContext, trainingDataPath);

            /*var minMaxEstimator1 = mlContext.Transforms.NormalizeMinMax("DiffPrev1");
            var minMaxEstimator2 = mlContext.Transforms.NormalizeMinMax("DiffPrev2");
            var minMaxEstimator3 = mlContext.Transforms.NormalizeMinMax("DiffPrev3");
            var minMaxEstimator4 = mlContext.Transforms.NormalizeMinMax("DiffPrev4");
            var minMaxEstimator5 = mlContext.Transforms.NormalizeMinMax("DiffPrev5");
            var minMaxEstimator6 = mlContext.Transforms.NormalizeMinMax("DiffPrev6");
            var minMaxEstimator7 = mlContext.Transforms.NormalizeMinMax("DiffPrev7");
            var minMaxEstimator8 = mlContext.Transforms.NormalizeMinMax("DiffPrev8");
            var minMaxEstimator9 = mlContext.Transforms.NormalizeMinMax("DiffPrev9");
            var minMaxEstimator10 = mlContext.Transforms.NormalizeMinMax("DiffPrev10");*/

            IDataView testDataView = LoadData(mlContext, testDataPath);

            //Testing if the data is loaded correctly
            Console.WriteLine(trainingDataView.Preview().RowView.ToString());
            //Can help: taxifare

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
                            .Append(mlContext.Transforms.Concatenate("Features", nameof(EuroDataModel.DiffPrev1), nameof(EuroDataModel.DiffPrev2), nameof(EuroDataModel.DiffPrev3), nameof(EuroDataModel.DiffPrev4), nameof(EuroDataModel.DiffPrev5),
                nameof(EuroDataModel.DiffPrev6), nameof(EuroDataModel.DiffPrev7), nameof(EuroDataModel.DiffPrev8), nameof(EuroDataModel.DiffPrev9), nameof(EuroDataModel.DiffPrev10)));

            //var trainingPipeline = mlContext.Transforms.Concatenate("Features", "DiffPrev1", "DiffPrev2", "DiffPrev3", "DiffPrev4", "DiffPrev5", "DiffPrev6", "DiffPrev7", "DiffPrev8", "DiffPrev9", "DiffPrev10")
            //   .Append(mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "Value", featureColumnName: "Features"));/*
            /*var trainer = mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "Value", featureColumnName: "Features");

            var trainingPipeline = mlContext.Transforms.Concatenate("Features", nameof(EuroDataModel.DiffPrev1), nameof(EuroDataModel.DiffPrev2), nameof(EuroDataModel.DiffPrev3), nameof(EuroDataModel.DiffPrev4), nameof(EuroDataModel.DiffPrev5),
                nameof(EuroDataModel.DiffPrev6), nameof(EuroDataModel.DiffPrev7), nameof(EuroDataModel.DiffPrev8), nameof(EuroDataModel.DiffPrev9), nameof(EuroDataModel.DiffPrev10))
                    .Append(mlContext.Transforms.CopyColumns(outputColumnName: "Value", inputColumnName: nameof(EuroDataModel.Value)))
                    .Append(trainer);*/

            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataprocessPipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(trainingDataView);



            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
            //var metrics = context.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number", scoreColumnName: "Score");
            Console.WriteLine("Confusion matrix: " + metrics.RSquared.ToString());

            mlContext.Model.Save(trainedModel, trainingDataView.Schema, networkPath);

            var loadedtrainedModel = mlContext.Model.Load(networkPath, out var modelInputSchema);
        

        // Create prediction engine related to the loaded trained model
        var predEngine = mlContext.Model.CreatePredictionEngine<EuroDataModel, EuroDataModelOutput>(loadedtrainedModel);


        //
        var resultprediction1 = predEngine.Predict(SampleEuroData.Euro1);
            MessageBox.Show($"Value was: 326.0800, the predicted value is:{resultprediction1.Score}");


    }
        #endregion

        public IDataView LoadData(MLContext context, string dataPath)
        {
            IDataView loadedData = null;
            //loading TrainingData
            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer))
            {
                loadedData = context.Data.LoadFromTextFile<EuroDataModel>(path: dataPath,
                       hasHeader: true,
                       separatorChar: ','
                       );
            }

            return loadedData;

        }

        public EstimatorChain<KeyToValueMappingTransformer> ConfigureNetwork(MLContext context)
        {
            // STEP 2: Common data process configuration with pipeline data transformations
            // Use in-memory cache for small/medium datasets to lower training time. Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.
            var dataProcessPipeline = context.Transforms.Conversion.MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(context.Transforms.Concatenate("Features", nameof(InputData.PixelValues))
                .AppendCacheCheckpoint(context));

            // STEP 3: Set the training algorithm, then create and config the modelBuilder
            var trainer = context.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline
                .Append(trainer)
                .Append(context.Transforms.Conversion.MapKeyToValue("Number", "Label"));

            return trainingPipeline;
        }
    }
}

