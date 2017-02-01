package br.unicamp.ic.execution;

public class TicTacFeatureSelectionExecution extends FeatureSelectionExecution {

	public TicTacFeatureSelectionExecution(boolean[] features) {
		super("tic-tac-toe.arff", features, 100, 6, 0.1);
	}

	public static void main(String[] args) {
		boolean[] features = { true, true, true, true, true, true, true, true,
				true };
		FeatureSelectionExecution fs = new TicTacFeatureSelectionExecution(
				features);
		fs.executeAll();
	}

}
