package br.unicamp.ic.execution.stacking;

public class DiabetesFeatureSelectionExecution extends FeatureSelectionExecution {

	public DiabetesFeatureSelectionExecution(boolean[] features) {
		super("diabetes.arff", features, 200, 6, 0.2);
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true};		
		FeatureSelectionExecution fs = new DiabetesFeatureSelectionExecution(features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesWithNoFilters();		
		executeWithNoFilter();
	}
	
}
