package br.unicamp.ic.execution.svm;

import weka.classifiers.functions.LibSVM;

public class WineFeatureSelectionExecution extends FeatureSelectionExecution {

	public WineFeatureSelectionExecution(boolean[] features) {
		super("wine.arff", features, 100, 6, 0.1, new LibSVM());
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,
				true, true, true, true, true};
		FeatureSelectionExecution fs = new WineFeatureSelectionExecution(features);
		fs.executeAll();
	}
	
}
