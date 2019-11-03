using App.Metrics;
using MachineLearningWithMetrics.Metrics;
using MachineLearningWithMetrics.MLdotNET;
using MachineLearningWithMetrics.MLdotNET.Predictors;
using MachineLearningWithMetrics.MLdotNET.Predictors.Common_Classes;
using MachineLearningWithMetrics.ViewModels.Commands;
using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Windows;
using System.Windows.Input;

namespace MachineLearningWithMetrics.ViewModels
{
    public class MainPageViewModel : INotifyPropertyChanged
    {

        #region Events, Fields and Properties

        public event PropertyChangedEventHandler PropertyChanged;

        private IPredictor selectedProblem;
        public IPredictor SelectedProblem
        {
            get
            {
                return this.selectedProblem;
            }
            set
            {
                this.selectedProblem = value;
            }
        }

        private Algorithms.BinaryClassificationTrainingAlgorithm selectedBinaryAlgorithm;
        public Algorithms.BinaryClassificationTrainingAlgorithm SelectedBinaryAlgorithm
        {
            get
            {
                return this.selectedBinaryAlgorithm;
            }
            set
            {
                this.selectedBinaryAlgorithm = value;
            }
        }

        private Algorithms.MultiClassificationTrainingAlgorithm selectedMultiAlgorithm;
        public Algorithms.MultiClassificationTrainingAlgorithm SelectedMultiAlgorithm
        {
            get
            {
                return this.selectedMultiAlgorithm;
            }
            set
            {
                this.selectedMultiAlgorithm = value;
            }
        }

        private Algorithms.RegressionTrainingAlgorithm selectedRegressionAlgorithm;
        public Algorithms.RegressionTrainingAlgorithm SelectedRegressionAlgorithm
        {
            get
            {
                return this.selectedRegressionAlgorithm;
            }
            set
            {
                this.selectedRegressionAlgorithm = value;
            }
        }
        private IMetricsRoot metrics;

        private Algorithms.BinaryClassificationTrainingAlgorithm binaryAlgorithms;
        public Algorithms.BinaryClassificationTrainingAlgorithm BinaryAlgorithms
        {
            get { return this.binaryAlgorithms; }
            set { this.binaryAlgorithms = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(binaryAlgorithms)));
            }
        }

        private Algorithms.MultiClassificationTrainingAlgorithm multiAlgorithms;
        public Algorithms.MultiClassificationTrainingAlgorithm MultiAlgorithms
        {
            get { return this.multiAlgorithms; }
            set
            {
                this.multiAlgorithms = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(MultiAlgorithms)));
            }
        }

        private Algorithms.RegressionTrainingAlgorithm regressionAlgorithms;
        public Algorithms.RegressionTrainingAlgorithm RegressionAlgorithms
        {
            get { return this.regressionAlgorithms; }
            set
            {
                this.regressionAlgorithms = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(RegressionAlgorithms)));
            }
        }

        #endregion

        #region Collections
        private ObservableCollection<IPredictor> problems = new ObservableCollection<IPredictor>();
        public ObservableCollection<IPredictor> Problems
        {
            get{return this.problems;}

            set
            {
                this.problems = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Problems)));
            }
        }

        private ObservableCollection<string> trainingAlgo = new ObservableCollection<string>();
        public ObservableCollection<string>  TrainingAlgo
        {
            get { return this.trainingAlgo; }

            set
            {
                this.trainingAlgo = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(TrainingAlgo)));
            }
        }

        private double[] trainingTestRate;

        public double[] TrainingTestRate
        {
            get { return trainingTestRate; }
            set { trainingTestRate = value;          
                
            }
        }

        private double selectedTrainingTestRate;

        public double SelectedTrainingTestRate
        {
            get { return selectedTrainingTestRate; }
            set
            {
                this.selectedTrainingTestRate = value;
            }
        }

        #endregion

        #region Constructors
        public MainPageViewModel()
        {
            InitializeMetrics();
            SetProblems();
            SetRate();
        }

        private void SetRate()
        {
            this.TrainingTestRate = new double[]
            {
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9
            };
        }
        #endregion


        #region Commands
        private ICommand _startInfluxCommand;
        private ICommand _startMLdotNETCommand;
        private ICommand _startGrafanaCommand;
        private ICommand _startMNISTCommand;
        private ICommand _startMNIST28Command;
        private ICommand _startEuroCommand;
        private ICommand _startWineQualityCommand;
        private ICommand _startBankNotesCommand;
        private ICommand _startNetworkCommand;

        public ICommand StartInfluxCommand
        {
            get
            {
                return _startInfluxCommand ?? (_startInfluxCommand = new CommandHandler(() => StartInflux(), () => CanExecute));
            }
        }

        public bool CanExecute
        {
            get
            {                
                return true;
            }
        }

        public void StartInflux()
        {
           string path = Environment.CurrentDirectory;
           Process.Start(path + @"/InfluxDB/influxd.exe");           
        }

        public void InitializeMetrics()
        {
            metrics = MetricsInitializer.Metrics;
            //MetricsInitializer metricsInitializer = new MetricsInitializer();
        }

        public ICommand StartMLdotNETCommand
        {
            get
            {
                return _startMLdotNETCommand ?? (_startMLdotNETCommand = new CommandHandler(() => StartMLdotNET(), () => CanExecute));
            }
        }       

        public void StartMLdotNET()
        {
            HeartPredictor heartPredictor = new HeartPredictor();
            this.MLdotNETPaths = heartPredictor.GetPath();
        }

        public ICommand StartGrafanaCommand
        {
            get
            {
                return _startGrafanaCommand ?? (_startGrafanaCommand = new CommandHandler(() => StartGrafana(), () => CanExecute));
            }
        }

        public void StartGrafana()
        {
            Process.Start("http://localhost:3000/d/RZBikTpWk/new-dashboard-copy?orgId=1");
        }

        public ICommand StartMNISTCommand
        {
            get
            {
                return _startMNISTCommand ?? (_startMNISTCommand = new CommandHandler(() => StartMNIST(), () => CanExecute));
            }
        }

        public void StartMNIST()
        {
            MNISTPredictor mnistPredictor = new MNISTPredictor();
        }

        public ICommand StartMNIST28Command
        {
            get
            {
                return _startMNIST28Command ?? (_startMNIST28Command = new CommandHandler(() => StartMNIST28(), () => CanExecute));
            }
        }

        public void StartMNIST28()
        {
            MNIST28Predictor mnist28Predictor = new MNIST28Predictor();
        }

        public ICommand StartEuroCommand
        {
            get
            {
                return _startEuroCommand ?? (_startEuroCommand = new CommandHandler(() => StartEuro(), () => CanExecute));
            }
        }

        public void StartEuro()
        {
            EuroPredictor euroPredictor = new EuroPredictor();
        }

        public ICommand StartWineQualityCommand
        {
            get
            {
                return _startWineQualityCommand ?? (_startWineQualityCommand = new CommandHandler(() => StartWineQualiy(), () => CanExecute));
            }
        }

        public void StartWineQualiy()
        {
            WineQualityPredictor wineQualityPredictor = new WineQualityPredictor();
        }

        public ICommand StartBankNotesCommand
        {
            get
            {
                return _startBankNotesCommand ?? (_startBankNotesCommand = new CommandHandler(() => StartBankNotes(), () => CanExecute));
            }
        }

        public void StartBankNotes()
        {
            BankNotePredictor bankNotePredictor = new BankNotePredictor();
        }

        public ICommand StartNetworkCommand
        {
            get
            {
                return _startNetworkCommand ?? (_startNetworkCommand = new CommandHandler(() => StartNetwork(), () => CanExecute));
            }
        }

        public void StartNetwork()
        {
            try
            {
                this.SelectedProblem.TrainTestDataRate = this.SelectedTrainingTestRate;
                switch (SelectedProblem)
                {
                    case BankNotePredictor bankNotePredictor:
                        SelectedProblem.SetAlgorithm(this.SelectedBinaryAlgorithm);
                        break;
                    case MNIST28Predictor mNIST28Predictor:
                        SelectedProblem.SetAlgorithm(this.SelectedMultiAlgorithm);
                        break;
                    case EuroPredictor euroPredictor:
                        SelectedProblem.SetAlgorithm(this.RegressionAlgorithms);
                        break;
                }
                this.SelectedProblem.ProcessNetwork();
            }
            catch(Exception e)
            {
                MessageBox.Show("Hiba");
            }
        }
        #endregion

        #region Properties
        private string _mLdotNETPaths = string.Empty;
       

        public string MLdotNETPaths
        {
            get { return _mLdotNETPaths; }
            set {
                _mLdotNETPaths = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(MLdotNETPaths)));
            }
        }

        #endregion

        #region Private methods
        private void SetProblems()
        {
            this.problems.Add(new BankNotePredictor());
            this.problems.Add(new MNIST28Predictor());
            this.problems.Add(new EuroPredictor());
        }
        #endregion
    }
}
