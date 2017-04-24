package com.vin;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.servlet.ServletContext;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Servlet implementation class ManaDeepImg
 */
@WebServlet("/ManaDeepImg")
public class ManaDeepImg extends HttpServlet {
	private static final long serialVersionUID = 1L;
       
    /**
     * @see HttpServlet#HttpServlet()
     */
    public ManaDeepImg() {
        super();
        // TODO Auto-generated constructor stub
    }

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// TODO Auto-generated method stub
		String fname = request.getParameter("file");
		fname = "/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/train/" + fname;
	    response.setHeader("Content-disposition","attachment; filename="+request.getParameter("file")+".png");
	    FileInputStream fis = new FileInputStream(new File(fname));  
	    final BufferedImage tif = ImageIO.read(fis);
	    ImageIO.write(tif, "png", response.getOutputStream());
	    response.getOutputStream().flush(); 
	    fis.close();
	}

	/**
	 * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// TODO Auto-generated method stub
		doGet(request, response);
	}

}
