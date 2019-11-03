using App.Metrics;
using MachineLearningWithMetrics.Metrics;
using MachineLearningWithMetrics.MLdotNET;
using MachineLearningWithMetrics.MLdotNET.Predictors;
using MachineLearningWithMetrics.ViewModels.Commands;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Windows.Input;

namespace MachineLearningWithMetrics.ViewModels
{
    public class MainPageViewModel : INotifyPropertyChanged
    {

        #region Events and Fields

        public event PropertyChangedEventHandler PropertyChanged;
        IMetricsRoot metrics;

        #endregion

        #region Constructors
        public MainPageViewModel()
        {
            InitializeMetrics();
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
    }
}
