try:
    import pandas as pd
    from tabulate import tabulate
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
import os
import csv

def view_history():
    history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_history.csv")
    
    if not os.path.exists(history_file):
        print("\n❌ No training history foundyet. Train a model first!")
        print(f"Expected path: {history_file}")
        return

    print("\n" + "═"*110)
    print("📜 TRAINING HISTORY LOG")
    print("═"*110)

    if HAS_LIBS:
        try:
            df = pd.read_csv(history_file)
            
            # Format duration to be more readable
            def format_duration(seconds):
                seconds = float(seconds)
                mins = int(seconds // 60)
                secs = int(seconds % 60)
                if mins > 0:
                    return f"{mins}m {secs}s"
                return f"{secs}s"
                
            df['Duration'] = df['Duration (s)'].apply(format_duration)
            
            # Select and reorder columns for display
            display_cols = ['Date', 'Time', 'Duration', 'Words (Classes)', 'Train Accuracy', 'Val Accuracy', 'Augmented']
            
            # Print with tabulate for a beautiful CLI table
            print(tabulate(df[display_cols], headers='keys', tablefmt='fancy_grid', showindex=False))
            
            print("\n" + "═"*110)
            print(f"📊 Total sessions recorded: {len(df)}")
            print(f"📂 History file: {history_file}")
            return
        except Exception as e:
            print(f"⚠️ Error with advanced formatting: {e}. Falling back to basic view...")

    # Fallback to basic CSV printing
    try:
        with open(history_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader)
            if not data:
                print("History file is empty.")
                return
            
            # Simple column alignment
            widths = [max(len(str(x)) for x in col) for col in zip(*data)]
            format_str = " | ".join(["{:<" + str(w) + "}" for w in widths])
            
            for i, row in enumerate(data):
                print(format_str.format(*row))
                if i == 0:
                    print("-" * (sum(widths) + 3*len(widths)))
    except Exception as e:
        print(f"❌ Error reading history file: {e}")

def view_comparison_history():
    comp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_history.csv")
    
    if not os.path.exists(comp_file):
        print("\n❌ No comparison history found. Run compare_models.py first!")
        return

    print("\n" + "═"*110)
    print("📈 COMPARISON SESSION LOG")
    print("═"*110)

    if HAS_LIBS:
        try:
            df = pd.read_csv(comp_file)
            def format_duration(seconds):
                seconds = float(seconds)
                mins = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            
            df['Total Dur'] = df['Total Duration (s)'].apply(format_duration)
            df['Base Dur'] = df['Baseline Dur (s)'].apply(format_duration)
            df['Aug Dur'] = df['Augmented Dur (s)'].apply(format_duration)
            
            display_cols = ['Date', 'Time', 'Total Dur', 'Base Dur', 'Aug Dur', 'Words', 'Baseline Acc', 'Augmented Acc', 'Improvement']
            print(tabulate(df[display_cols], headers='keys', tablefmt='fancy_grid', showindex=False))
            print("\n" + "═"*110)
            return
        except Exception as e:
            print(f"⚠️ Error formatting comparison history: {e}")

    # Fallback
    try:
        with open(comp_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                print(" | ".join(row))
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--comp":
        view_comparison_history()
    else:
        view_history()
        print("\n💡 Tip: Use 'python view_history.py --comp' to see comparison session history!")
