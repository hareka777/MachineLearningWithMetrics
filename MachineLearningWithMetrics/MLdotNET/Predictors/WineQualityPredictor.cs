using App.Metrics;
using MachineLearningWithMetrics.Metrics;
using MachineLearningWithMetrics.MLdotNET.DataModel.WineQuality;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using static MachineLearningWithMetrics.MLdotNET.Predictors.Common_Classes.Algorithms;

namespace MachineLearningWithMetrics.MLdotNET.Predictors
{
    //DataSource: https://machinelearningmastery.com/standard-machine-learning-datasets/

    public class WineQualityPredictor : IPredictor
    {
        #region Fields
        private static readonly string dataFolderPath = Paths.dataFolderPath + @"\WineQuality";
        private readonly string dataModelFolderPath = Paths.dataModelFolderPath;
        private readonly string networkPath = Paths.networkModelFolderPath + @"\WineQuality.zip";

        private string trainingDataPath = dataFolderPath + @"\winequality-white.csv";
        private string testDataPath = dataFolderPath + @"\winequality-white.csv";

        MultiClassificationTrainingAlgo trainingAlgo;

        IMetricsRoot _metrics;
        #endregion

        #region Constructors

        public WineQualityPredictor()
        {
            this._metrics = MetricsInitializer.Metrics;
            trainingAlgo = MultiClassificationTrainingAlgo.NaiveBayes;
            ProcessNetwork();
        }

        public WineQualityPredictor(MultiClassificationTrainingAlgo trainingAlgo)
        {
            this._metrics = MetricsInitializer.Metrics;
            this.trainingAlgo = trainingAlgo;
            ProcessNetwork();
        }

        #endregion

        #region Implemeting Abstact Methods

        internal override void ProcessNetwork()
        {
            EstimatorChain<ITransformer> pipeline = null;

            try
            {
                Dictionary<string, string> tags = new Dictionary<string, string>();
                tags.Add("ProcessName", "LoadingData");
                tags.Add("TaskName", "WineQuality");

                var metricsTags = new MetricTags(tags.Keys.ToArray(), tags.Values.ToArray());

                using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, metricsTags))
                {
                    trainingData = LoadData(mlContext, trainingDataPath);
                    testData = LoadData(mlContext, testDataPath);
                }

                Console.WriteLine(trainingData.Preview().RowView.ToString());

                //MessageBox.Show("Data successfully loaded!");
            }
            catch (Exception e)
            {
                MessageBox.Show("Could not load data: " + e.Message);
            }
            try
            {
                pipeline = ConfigureNetwork();
            }
            catch (Exception e)
            {
                MessageBox.Show("Could not set pipeline: " + e.Message);
            }
            try
            {
                using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer))
                {
                    trainedModel = Train(pipeline);
                }
            }
            catch (Exception e)
            {
                MessageBox.Show("Training was not successful: " + e.Message);
            }
        }
        

        internal override void EvaluateModel(ITransformer trainedModel)
        {
            throw new System.NotImplementedException();
        }

        internal override IDataView LoadData(MLContext context, string dataPath)
        {
            IDataView loadedData = null;
            //loading TrainingData
            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer))
            {
                loadedData = context.Data.LoadFromTextFile<WineQualityInput>(path: dataPath,                       
                        hasHeader: true,
                        separatorChar: ';'
                        );
            }

            return loadedData;
        }        

        internal override void SaveNetwork(ITransformer trainedModel)
        {
            throw new System.NotImplementedException();
        }        
        #endregion

        #region Private Methods
        private EstimatorChain<ITransformer> ConfigureNetwork()
        {
            var dataPrepration = mlContext.Transforms.Concatenate("Features", "FixedAcidity", "VolatileAcidity", "CriticAcid", "ResidualSugar",
                "Chlorides", "FreeSulfurDioxide", "TotalSulfurDioxide", "Density", "Exang", "Sulphates", "Alcohol")
                //Normalizing input parameters
                .Append(mlContext.Transforms.NormalizeLogMeanVariance(outputColumnName: nameof(WineQualityInput.FixedAcidity)))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance(outputColumnName: nameof(WineQualityInput.VolatileAcidity)))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance(outputColumnName: nameof(WineQualityInput.CriticAcid)))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance(outputColumnName: nameof(WineQualityInput.ResidualSugar)))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance(outputColumnName: nameof(WineQualityInput.Chlorides)))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance(outputColumnName: nameof(WineQualityInput.FreeSulfurDioxide)))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance(outputColumnName: nameof(WineQualityInput.TotalSulfurDioxide)))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance(outputColumnName: nameof(WineQualityInput.Density)))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance(outputColumnName: nameof(WineQualityInput.Exang)))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance(outputColumnName: nameof(WineQualityInput.Sulphates)))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance(outputColumnName: nameof(WineQualityInput.Alcohol)));

            // STEP 3: Set the training algorithm, then create and config the modelBuilder
            var trainer = ApplyTrainingAlgo(mlContext, trainingAlgo);


            var trainingPipeline = dataPrepration
                .Append(trainer);

            return trainingPipeline;
        }

        private ITransformer Train(EstimatorChain<ITransformer> pipeline)
        {
            return pipeline.Fit(trainingData);
        }

        internal override void TestSomePredictions()
        {
            throw new NotImplementedException();
        }
        #endregion
    }
}
