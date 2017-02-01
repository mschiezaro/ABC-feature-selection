package br.unicamp.ic.util;

import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class LogUtil {

	private FileHandler fileHandler;

	private Logger logger;

	public LogUtil() {
		try {
			fileHandler = new FileHandler("feature_selection.log");
			fileHandler.setFormatter(new SimpleFormatter());
		} catch (SecurityException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public Logger getLogger() {
		if (logger == null) {
			logger = Logger.getLogger("br.unicamp.ic.featureselection");
			logger.addHandler(fileHandler);
			logger.setLevel(Level.ALL);			
		}
		return logger;
	}
}
