using MachineLearningWithMetrics.ViewModels;
using System.Windows;

namespace MachineLearningWithMetrics
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            DataContext = new MainPageViewModel();
        }
    }
}
