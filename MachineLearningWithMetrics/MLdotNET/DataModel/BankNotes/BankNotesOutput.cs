using Microsoft.ML.Data;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.BankNotes
{
    /*
     * Data model class for binary classification
     * Network output class
     */
    public class BankNotesOutput    {

        [ColumnName("PredictedLabel")]
        public bool Prediction;

        [ColumnName("Probability")]
        public float Probability;

        [ColumnName("Score")]
        public float Score;
    }
}
