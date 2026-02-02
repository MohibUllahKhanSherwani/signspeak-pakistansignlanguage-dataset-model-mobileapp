import csv
import datetime
import os

def log_training_session(duration_seconds, num_words, training_acc, val_acc, epochs, batch_size, augmented, model_path="action_model.h5"):
    """
    Logs the training session details to a CSV file for history tracking and comparison.
    """
    # Get the directory of this script to store the CSV in the same place
    current_dir = os.path.dirname(os.path.abspath(__file__))
    history_file = os.path.join(current_dir, "training_history.csv")
    
    file_exists = os.path.isfile(history_file)
    
    try:
        with open(history_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # If file doesn't exist, write the header
            if not file_exists:
                writer.writerow([
                    "Date", 
                    "Time", 
                    "Duration (s)", 
                    "Words (Classes)", 
                    "Train Accuracy", 
                    "Val Accuracy", 
                    "Epochs", 
                    "Batch Size", 
                    "Augmented", 
                    "Model Saved As"
                ])
            
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            writer.writerow([
                date_str, 
                time_str, 
                round(duration_seconds, 2), 
                num_words, 
                f"{training_acc:.4f}" if training_acc is not None else "N/A", 
                f"{val_acc:.4f}" if val_acc is not None else "N/A", 
                epochs, 
                batch_size, 
                "Yes" if augmented else "No",
                model_path
            ])
        
        print(f"📊 Training session recorded in: {history_file}")
    except Exception as e:
        print(f"⚠️ Warning: Could not log training session: {e}")

def log_comparison_session(total_duration, baseline_duration, augmented_duration, num_words, baseline_acc, augmented_acc):
    """
    Logs the detailed breakdown of a comparison run.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    comp_file = os.path.join(current_dir, "comparison_history.csv")
    
    file_exists = os.path.isfile(comp_file)
    
    try:
        with open(comp_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    "Date", "Time", "Total Duration (s)", "Baseline Dur (s)", 
                    "Augmented Dur (s)", "Words", "Baseline Acc", 
                    "Augmented Acc", "Improvement"
                ])
            
            now = datetime.datetime.now()
            improvement = augmented_acc - baseline_acc
            
            writer.writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                round(total_duration, 2),
                round(baseline_duration, 2),
                round(augmented_duration, 2),
                num_words,
                f"{baseline_acc:.4f}",
                f"{augmented_acc:.4f}",
                f"{improvement:+.4f}"
            ])
        print(f"📈 Comparison session recorded in: {comp_file}")
    except Exception as e:
        print(f"⚠️ Warning: Could not log comparison session: {e}")
