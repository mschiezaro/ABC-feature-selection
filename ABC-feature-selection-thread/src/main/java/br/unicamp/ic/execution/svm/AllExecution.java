package br.unicamp.ic.execution.svm;

import java.io.BufferedWriter;
import java.io.IOException;

import br.unicamp.ic.util.FileUtil;

public class AllExecution {

	protected BufferedWriter writer;

	public AllExecution() {
		writer = FileUtil.newInstance().getWriter();
	}

	public static void main(String[] args) {

		AllExecution e = new AllExecution();

		e.writeMsg("AutosFeatureSelectionExecution");
		boolean features[] = { true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true };
		FeatureSelectionExecution fs = new AutosFeatureSelectionExecution(
				features);
		fs.executeAll();

		e.writeMsg("BCancerFeatureSelectionExecution");
		features = new boolean[] { true, true, true, true, true, true, true,
				true, true };
		fs = new BCancerFeatureSelectionExecution(features);
		fs.executeAll();

		e.writeMsg("DiabetesFeatureSelectionExecution");
		features = new boolean[] { true, true, true, true, true, true, true,
				true };
		fs = new DiabetesFeatureSelectionExecution(features);
		fs.executeAll();

		e.writeMsg("GlassFeatureSelectionExecution");
		features = new boolean[] { true, true, true, true, true, true, true,
				true, true };
		fs = new GlassFeatureSelectionExecution(features);
		fs.executeAll();

		e.writeMsg("HeartCFeatureSelectionExecution");
		features = new boolean[] { true, true, true, true, true, true, true,
				true, true, true, true, true, true };
		fs = new HeartCFeatureSelectionExecution(features);
		fs.executeAll();

		e.writeMsg("HeartStatlogFeatureSelectionExecution");
		features = new boolean[] { true, true, true, true, true, true, true,
				true, true, true, true, true, true };
		fs = new HeartStatlogFeatureSelectionExecution(features);
		fs.executeAll();

		e.writeMsg("HepaticFeatureSelectionExecution");
		features = new boolean[] { true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true };
		fs = new HepaticFeatureSelectionExecution(features);
		fs.executeAll();

		e.writeMsg("IrisFeatureSelectionExecution");
		features = new boolean[] { true, true, true, true };
		fs = new IrisFeatureSelectionExecution(features);
		fs.executeAll();

		e.writeMsg("LaborFeatureSelectionExecution");
		features = new boolean[] { true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true };
		fs = new LaborFeatureSelectionExecution(features);
		fs.executeAll();
		
		e.writeMsg("ImageSegmentationFeatureSelectionExecution");
		features = new boolean[] { true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true };
		fs = new ImageSegmentationFeatureSelectionExecution(features);
		fs.executeAll();
		
		FileUtil.newInstance().close();
	}

	protected void writeMsg(String msg) {
		try {
			writer.newLine();
			writer.newLine();
			writer.write(msg);
			writer.newLine();
			writer.flush();
		} catch (IOException e) {
			throw new RuntimeException();
		}
	}
}
