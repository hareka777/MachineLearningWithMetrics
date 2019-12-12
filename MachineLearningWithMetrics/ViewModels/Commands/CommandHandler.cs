using System;
using System.Windows.Input;

namespace MachineLearningWithMetrics.ViewModels.Commands
{
    public class CommandHandler : ICommand
    {
        private Action _action;
        private Func<bool> _canExecute;


        public CommandHandler(Action action, Func<bool> canExecute)
        {
            _action = action;
            _canExecute = canExecute;
        }

        /*
         * Detects if the CanExecute parameter changed
         */
        public event EventHandler CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }

        public bool CanExecute(object parameter)
        {
            return _canExecute.Invoke();
        }

        public void Execute(object parameter)
        {
            _action();
        }
    }
}
