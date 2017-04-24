<%@page import="java.io.File"%>
<%@page import="java.io.FileInputStream"%>
<%

	String fname = request.getParameter("file");
	fname = "/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/train/" + fname;
	FileInputStream fos = new FileInputStream(new File(fname));
	byte []b = new byte[1024];
	int len = fos.read(b);
	response.setContentType("jpeg");
	while(len>0){
		response.getOutputStream().write(b, 0, len);		
		len = fos.read(b);
	}
	fos.close();
%>