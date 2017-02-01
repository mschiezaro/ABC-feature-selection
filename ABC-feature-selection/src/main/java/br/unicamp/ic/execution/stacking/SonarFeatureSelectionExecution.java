package br.unicamp.ic.execution.stacking;

public class SonarFeatureSelectionExecution extends FeatureSelectionExecution {

	public SonarFeatureSelectionExecution(boolean[] features) {
		super("sonar.arff", features, 100, 6, 0.1);
	}

	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true };
		FeatureSelectionExecution fs = new SonarFeatureSelectionExecution(features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesWithNoFilters();		
		executeWithNoFilter();
	}
}
