using Microsoft.ML.Data;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.WineQuality
{
    public class WineQualityInput
    {
        //Fixed acidity
        [LoadColumn(0)]
        public float FixedAcidity { get; set; }

        //Volatile acidity
        [LoadColumn(1)]
        public float VolatileAcidity { get; set; }

        //Citric acid
        [LoadColumn(2)]
        public float CriticAcid { get; set; }

        //Residual sugar
        [LoadColumn(3)]
        public float ResidualSugar { get; set; }

        //Chlorides
        [LoadColumn(4)]
        public float Chlorides { get; set; }

        //Free sulfur dioxide
        [LoadColumn(5)]
        public float FreeSulfurDioxide { get; set; }

        //Total sulfur dioxide
        [LoadColumn(6)]
        public float TotalSulfurDioxide { get; set; }

        //Density
        [LoadColumn(7)]
        public float Density { get; set; }

        //pH
        [LoadColumn(8)]
        public float Exang { get; set; }

        //Sulphates
        [LoadColumn(9)]
        public float Sulphates { get; set; }

        //Alcohol
        [LoadColumn(10)]
        public float Alcohol { get; set; }

        //Quality(score between 0 and 10)
        [LoadColumn(11)]
        public int Quality { get; set; }
    }
}
