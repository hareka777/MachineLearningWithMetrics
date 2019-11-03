namespace MachineLearningWithMetrics.MLdotNET.DataModel.BankNotes
{
    public class SampleBankNotesDatacs
    {
        #region Private Fields
        private static BankNotesInput authentic1 = new BankNotesInput()
        {
            Variance = -4.1958f,
            Skewness = -8.1819f,
            Kurtosis = 12.1291f,
            Entropy = -1.6017f
        };

        private static BankNotesInput authentic2 = new BankNotesInput()
        {
            Variance = 1.5077f,
            Skewness = 1.9596f,
            Kurtosis = -3.0584f,
            Entropy = -0.12243f
        };

        private static BankNotesInput inauthentic1 = new BankNotesInput()
        {
            Variance = 5.2423f,
            Skewness = 11.0272f,
            Kurtosis = -4.353f,
            Entropy = -4.1013f
        };

        private static BankNotesInput inauthentic2 = new BankNotesInput()
        {
            Variance = 4.0102f,
            Skewness = 10.6568f,
            Kurtosis = -4.1388f,
            Entropy = -5.0646f
        };
        #endregion

        #region Public Properties

        public static  BankNotesInput Authentic1
        {
            get { return authentic1; }  
        }

        public static BankNotesInput Authentic2
        {
            get { return authentic2; }
        }

        public static BankNotesInput InAuthentic1
        {
            get { return inauthentic1; }
        }

        public static BankNotesInput InAuthentic2
        {
            get { return inauthentic2; }
        }


        #endregion
    }
}