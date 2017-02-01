package br.unicamp.ic.util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class FileUtil {
	
	private static FileUtil fileUtil = null;
	private File file;
	private BufferedWriter bw;
	
	private FileUtil() {
		file = new File("feature_selection.log");
		if (file.exists()) {
			file.delete();
		}
		try {
			file.createNewFile();
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			bw = new BufferedWriter(fw);		
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	private FileUtil(String filename) {
		file = new File(filename);
		if (file.exists()) {
			file.delete();
		}
		try {
			file.createNewFile();
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			bw = new BufferedWriter(fw);		
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	
	public static final FileUtil newInstance() {
		if (fileUtil == null) {
			fileUtil = new FileUtil();
		}
		return fileUtil;
		
	}
	
	public static final FileUtil newInstance(String filename) {
		return  new FileUtil(filename);
	}
	
	
	public BufferedWriter getWriter() {
		return bw;
	}
	
	public void flush() {
		try {
			bw.flush();
		} catch (IOException e) {
			throw new RuntimeException();
		}
	}

	public void close() {
		try {
			bw.close();
		} catch (IOException e) {
			throw new RuntimeException();
		}
	}
	
}
