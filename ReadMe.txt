Data
Ensure the Counsel Chat dataset is downloaded and placed in the project directory. The dataset should be named counsel_chat.csv.

Usage
Follow the steps below to run the scripts in the correct order to perform the complete analysis:

1)emp-rating.py
	Need to use Empathy Marker contains in EmpathyDataset.csv which train our model
	Will use counsel_chat.csv as input to test our model and will add new column with name empathy rating in update consuel file.
	It will also try to predict some other input to see is that empathatic or not.
	Usage: python emp-rating.py
	Outputs: Empathy ratings which will be used by subsequent scripts.
EmpathyScoreFormula.py
	Calculates empathy scores from the rated interactions.
	Usage: python EmpathyScoreFormula.py
	Outputs: Calculated empathy scores, which will be used for further analysis.
ComparisonChatgptVsOur.py
	Compares the empathy ratings between ChatGPT and our model using the calculated empathy scores.
	Usage: python ComparisonChatgptVsOur.py
	Outputs: A detailed comparison report highlighting the differences in performance.
ResultRepresentation.py
	Visualizes the results of the comparison in various graphical formats.
	Usage: python ResultRepresentation.py
	Outputs: Graphs and charts representing the analysis results.
Results
After running all scripts, the results will be available in the form of printed outputs and saved visualizations in the project directory.