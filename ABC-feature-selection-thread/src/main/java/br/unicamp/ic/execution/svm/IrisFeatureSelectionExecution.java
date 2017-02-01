package br.unicamp.ic.execution.svm;

import weka.classifiers.functions.LibSVM;
import br.unicamp.ic.util.FileUtil;

public class IrisFeatureSelectionExecution extends FeatureSelectionExecution {

	public IrisFeatureSelectionExecution(boolean[] features) {
		super("iris.arff", features, 20, 6, 0.1, new LibSVM());
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true};		
		FeatureSelectionExecution fs = new IrisFeatureSelectionExecution(features);
		fs.executeAll();
		FileUtil.newInstance().close();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesWithNoFilters();		
		executeWithNoFilter();
	}
}
