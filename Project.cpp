#include <royale.hpp>
#include <iostream>
#include <string>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <Windows.h>
#include <time.h>

std::string Eingabe = "1";



class MyListener : public royale::IDepthDataListener
{

public:

	void onNewData(const royale::DepthData *data)
	{
		// this callback function will be called for every new depth frame

		std::lock_guard<std::mutex> lock(flagMutex);
		zImage.create(cv::Size(data->width, data->height), CV_32FC1);
		zImage = 0;
		grayImage.create(cv::Size(data->width, data->height), CV_32FC1);
		grayImage = 0;
		int k = 0;
		for (int y = 0; y < zImage.rows; y++)
		{
			for (int x = 0; x < zImage.cols; x++)
			{
				auto curPoint = data->points.at(k);
				if (curPoint.depthConfidence > 0)
				{
					// if the point is valid
					zImage.at<float>(y, x) = curPoint.z;
					grayImage.at<float>(y, x) = curPoint.grayValue;
				}
				k++;
			}
		}

		cv::Mat temp = zImage.clone();
		undistort(temp, zImage, cameraMatrix, distortionCoefficients);
		temp = grayImage.clone();
		undistort(temp, grayImage, cameraMatrix, distortionCoefficients);

		//Kontrastspreizung
		double gmin, gmax;
		cv::minMaxLoc(grayImage, &gmin, &gmax);
		cv::convertScaleAbs(grayImage, grayImageMask, 255.0 / (gmax - gmin), -gmin * 255.0 / (gmax - gmin));

		cv::compare(zImage, 0, zImageMask, cv::CMP_GT);
		double zmin, zmax;
		cv::minMaxLoc(zImage, &zmin, &zmax);
		cv::convertScaleAbs(zImage, zImageMask, 255.0 / (zmax - zmin), -zmin * 255.0 / (zmax - zmin));



		//Colormap Tiefenbild
		applyColorMap(zImageMask, zImageColor, 4);

		//Bildanzeige
		cv::namedWindow("Grauwertbild", 1);
		imshow("Grauwertbild", grayImageMask);
		cv::namedWindow("Tiefenbild", 1);
		imshow("Tiefenbild", zImageColor);
		cv::waitKey(20);

		//Mittelwertfilter (20 Frames)
		if (!MittelwertInit) {
			MittelwertInit = true;
			cv::blur(grayImageMask, DynMittFilter, cv::Size(3, 3));
			DynMittFilterSumme = DynMittFilter.clone();
			DynMittFilterSumme.convertTo(DynMittFilterSumme, CV_32F);
			DynMittFilter.convertTo(DynMittFilter, CV_32F);
			DynMittFilterSumme = 0;
		}
		else
		{
			cv::blur(grayImageMask, DynMittFilter, cv::Size(3, 3));
			DynMittFilterSumme.convertTo(DynMittFilterSumme, CV_32F);
			DynMittFilter.convertTo(DynMittFilter, CV_32F);
		}

		DynMittFilterSumme = (DynMittFilterSumme + DynMittFilter);
		Mittelwertzaehler++;

		if (Mittelwertzaehler != 19) {
			DynMittFilter.convertTo(DynMittFilter, CV_8UC1);
		}
		else if (Mittelwertzaehler == 19) {
			DynMittFilterSumme /= 20;
			DynMittFilterSumme.convertTo(DynMittFilterSumme, CV_8UC1);
			DynMittFilter.convertTo(DynMittFilter, CV_8UC1);
			cv::namedWindow("Mittelwertfilter (20 Frames)", 1);
			imshow("Mittelwertfilter (20 Frames)", DynMittFilterSumme);
			Mittelwertzaehler = 0;
		}

		//Mittelwertfilter (1 Frame)
		for (int i = 1; i < 5; i = i + 2)
		{
			cv::blur(grayImageMask, MittFilter, cv::Size(i, i), cv::Point(-1, -1));

		}
		cv::namedWindow("Mittelwertfilter (1 Frame)", 1);
		imshow("Mittelwertfilter (1 Frame)", MittFilter);

		//Medianfilter
		for (int i = 1; i < 5; i = i + 2)
		{
			cv::medianBlur(grayImageMask, MedFilter, i);
		}

		cv::namedWindow("Medianfilter", 1);
		imshow("Medianfilter", MedFilter);

		//Linienprofil
		Linprof = cv::Mat::zeros(DynMittFilter.rows, DynMittFilter.cols, CV_8UC3);
		int Summe = 0;
		int Zwischensumme = 0;

		for (int i = 0; i < DynMittFilter.rows; i++)
		{
			for (int e = 0; e < DynMittFilter.cols; e++)
			{
				Summe += DynMittFilter.at<char>(i, e);
			}
			Summe /= DynMittFilter.cols;
			cv::line(Linprof, cv::Point((int)(DynMittFilter.cols / 2 + Zwischensumme), i), cv::Point((int)(DynMittFilter.cols / 2 + Summe), i + 1), cv::Scalar(0, 0, 255));
			Zwischensumme = Summe;
			Summe = 0;
		}

		for (int i = 0; i < MittFilter.rows; i++)
		{
			for (int e = 0; e < MittFilter.cols; e++)
			{
				Summe += MittFilter.at<char>(i, e);
			}
			Summe /= MittFilter.cols;
			cv::line(Linprof, cv::Point((int)(DynMittFilter.cols / 2 + Zwischensumme), i), cv::Point((int)(DynMittFilter.cols / 2 + Summe), i + 1), cv::Scalar(0, 255, 0));
			Zwischensumme = Summe;
			Summe = 0;
		}

		for (int i = 0; i < MedFilter.rows; i++)
		{
			for (int e = 0; e < MedFilter.cols; e++)
			{
				Summe += MedFilter.at<char>(i, e);
			}
			Summe /= MedFilter.cols;
			cv::line(Linprof, cv::Point((int)(DynMittFilter.cols / 2 + Zwischensumme), i), cv::Point((int)(DynMittFilter.cols / 2 + Summe), i + 1), cv::Scalar(255, 0, 0));
			Zwischensumme = Summe;
			Summe = 0;
		}

		cv::namedWindow("linienprofil", 1);
		imshow("linienprofil", Linprof);

		//Segmentierung
		//Schwellwertsegmentierung
		cv::adaptiveThreshold(MedFilter, Schwellwert, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 21, 21);
		cv::namedWindow("Schwellwert", 1);
		imshow("Schwellwert", Schwellwert);
		//Connected Components
		cv::connectedComponents(Schwellwert, ConnectionAr, 4, CV_16U);
		ConnectionAr.convertTo(ConnectionAr, CV_8UC3);
		cv::namedWindow("Komponente", 1);
		imshow("Komponente", ConnectionAr);
		//Farbcodierung
		cv::cvtColor(ConnectionAr, ColorLabel, cv::COLOR_GRAY2RGB);
		for (int i = 0; i <= 9; i++)
		{
			for (int r = 0; r < ConnectionAr.rows; r++)
			{
				for (int c = 0; c < ConnectionAr.cols; c++) {
					if (ConnectionAr.at<char>(r, c) == i)
						ColorLabel.at<cv::Vec3b>(r, c) = Farbpallete[i];
				}
			}
		}
		cv::namedWindow("Farblabel", 1);
		imshow("Farblabel", ColorLabel);

		//Tiefenbildauswertung
		//Histogramm
		cv::calcHist(&zImageMask, 1, 0, cv::Mat(), Histogramm, 1, &histSize, &histRange, true, false);
		Histogrammdarstellung = cv::Mat::zeros(50, 256, CV_8UC1);
		for (int i = 0; i < 255; i++) {
			cv::line(Histogrammdarstellung, cv::Point((int)i, 0), cv::Point((int)i, Histogramm.at<float>(i, 0)), cv::Scalar(255, 255, 255));
		}
		cv::namedWindow("Histogramm", 1);
		imshow("Histogramm", Histogrammdarstellung);

		//Histogramm-Glättung
		ksize = { 3,3 };
		cv::GaussianBlur(Histogramm, HistogrammGauss, ksize, 1, 0, cv::BORDER_DEFAULT);

		HistogrammdarstellungGauss = cv::Mat::zeros(50, 256, CV_8UC1);
		for (int i = 0; i < 255; i++) {
			cv::line(HistogrammdarstellungGauss, cv::Point((int)i, 0), cv::Point((int)i, HistogrammGauss.at<float>(i, 0)), cv::Scalar(255, 255, 255));
		}
		cv::namedWindow("HistogrammGauss", 1);
		imshow("HistogrammGauss", HistogrammdarstellungGauss);

		//Lokales Maximum
		MaxIndex = 0;
		for (int i = 0; i < 255; i++) {
			if (MaxIndex < HistogrammGauss.at<float>(i, 0))
				MaxIndex = i;
		}
		//Lokales Minimum
		Min = MaxIndex;
		for (int i = MaxIndex; i > 0; i--) {
			if (Min >  HistogrammGauss.at<float>(i, 0))
				Min = i;
			if (MinZwischen<Min) {
				break;
			}
			MinZwischen = Min;
		}
		//Binärbilderzeugung
		cv::threshold(zImageMask, BinaryBild, (double)Min - 20, 255, cv::THRESH_BINARY);

		cv::namedWindow("Binärbild", 1);
		imshow("Binärbild", BinaryBild);

		//Vergleichen des Binärbildes
		if (BinariActive) {
			cv::compare(BinaryBildZwischenspeicher, BinaryBild, BinaryBildVergleichGrößer, cv::CMP_GT);
			cv::morphologyEx(BinaryBildVergleichGrößer, BinaryBildVergleichGrößer, cv::MORPH_OPEN, cv::getStructuringElement(1, cv::Size(3, 3)));
			cv::namedWindow("Größer Binär", 1);
			imshow("Größer Binär", BinaryBildVergleichGrößer);

			cv::compare(BinaryBildZwischenspeicher, BinaryBild, BinaryBildVergleichKleiner, cv::CMP_LT);
			cv::morphologyEx(BinaryBildVergleichKleiner, BinaryBildVergleichKleiner, cv::MORPH_OPEN, cv::getStructuringElement(1, cv::Size(3, 3)));
			cv::namedWindow("Kleiner Binär", 1);
			imshow("Kleiner Binär", BinaryBildVergleichKleiner);


		}
		BinaryBild.copyTo(BinaryBildZwischenspeicher);
		BinariActive = true;

		//Farbvergleich

		BinaryBildSchwarz = cv::Mat::zeros(BinaryBildVergleichGrößer.rows, BinaryBildVergleichGrößer.cols, CV_8UC1);
		std::vector<cv::Mat> Merger;
		Merger.push_back(BinaryBildSchwarz);
		Merger.push_back(BinaryBildVergleichGrößer);
		Merger.push_back(BinaryBildVergleichKleiner);

		cv::merge(Merger, Farbvergleich);

		cv::namedWindow("Farbvergleich", 1);
		imshow("Farbvergleich", Farbvergleich);

		//Tastenzuweisung
		if (EternalColor)
			ColorLabel.copyTo(EwigeFarbpallete);

		cv::namedWindow("EwigeFarbpallete", 1);
		imshow("EwigeFarbpallete", EwigeFarbpallete);

		if (FreezeZaehler > 10) {
			EternalColor = false;
		}
		else
		{
			FreezeZaehler++;
		}

		//Fingererkennung

		for (int i = 0; BinaryBildVergleichKleiner.cols > i; i++) {
			for (int e = 0; BinaryBildVergleichKleiner.rows > e; e++) {

				if (BinaryBildVergleichKleiner.at<char>(e, i) != 0) {

					for (int r = 0; r < 10; r++) {
						if (EwigeFarbpallete.at<cv::Vec3b>(e, i) == Farbpallete[r]) {
							Anschlagzaehler[r]++;

						}
					}

				}

			}

		}


		//Tastaturanschlag

		for (int i = 2; i < 10; i++) {
			if (Anschlag < Anschlagzaehler[i])
				Anschlag = i;
		}


		std::cout << Buchstabenpallette[Anschlag] << std::endl;
		Ausgabe = (Buchstabenpallette[Anschlag]);
		cv::putText(Farbvergleich, Ausgabe, cv::Point((int)Farbvergleich.rows / 2, Farbvergleich.cols / 2), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Vec3b(255, 255, 255), 5);

		cv::namedWindow("Ausgabe", 1);
		imshow("Ausgabe", Farbvergleich);


		for (int r = 0; r < 10; r++) {
			Anschlagzaehler[r] = 0;
		}
		Anschlag = 0;


		//Speicher
		if (arg) {
			Video_Speichern();
		}
		cv::waitKey(20);


	}

	void Video_Anlegen(cv::String name, int hoehe, int breite, int framerate) {
		cv::String name_gray = name + "_gray.avi";
		cv::String name_depth = name + "_deth.avi";
		Video_gray.open(name_gray, cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), framerate, cv::Size(breite, hoehe), false);
		Video_depth.open(name_depth, cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), framerate, cv::Size(breite, hoehe), true);
	}

	void Video_Speichern() {
		Video_gray.write(grayImageMask);
		Video_depth.write(zImageColor);

	}


	int Anschlagzaehler[10] = { 0,0,0,0,0,0,0,0,0,0 };
	bool EternalColor = true;
	bool BinariActive = false;
	bool arg = false;
	bool MittelwertInit = false;
	int Mittelwertzaehler = 0;
	int FreezeZaehler = 0;
	int MaxIndex, Min, MinZwischen, Anschlag;
	cv::VideoWriter Video_gray, Video_depth;
	cv::Mat zImageColor, zImageMask, grayImageMask, MedFilter, MittFilter, DynMittFilter, DynMittFilterSumme, Linprof, Schwellwert, ConnectionAr, ColorLabel, Histogramm, Histogrammdarstellung, HistogrammGauss, HistogrammdarstellungGauss, BinaryBild, BinaryBildZwischenspeicher, BinaryBildVergleich, BinaryBildVergleichGrößer, BinaryBildVergleichKleiner,
		BinaryBildSchwarz, Farbvergleich, EwigeFarbpallete;
	cv::Scalar_<uint8_t> RGBPixel;
	cv::Vec3b Farbpallete[10] = { cv::Vec3b(0, 0, 0) , cv::Vec3b(0, 0, 0), cv::Vec3b(100, 149, 237), cv::Vec3b(47, 79, 79), cv::Vec3b(152, 251, 152), cv::Vec3b(240, 230, 140), cv::Vec3b(205, 92, 92), cv::Vec3b(233, 150, 122), cv::Vec3b(153, 50, 204), cv::Vec3b(205, 181, 205) };
	char Buchstabenpallette[10] = { ' ',' ','A','B','C','D','E','F','G','H' };
	int histSize = 256;
	float range[2] = { 0, 256 };
	const float* histRange = { range };
	cv::Size ksize;
	std::string Ausgabe;

	void setLensParameters(const royale::LensParameters &lensParameters)
	{
		// Construct the camera matrix
		// (fx   0    cx)
		// (0    fy   cy)
		// (0    0    1 )
		cameraMatrix = (cv::Mat1d(3, 3) << lensParameters.focalLength.first, 0, lensParameters.principalPoint.first,
			0, lensParameters.focalLength.second, lensParameters.principalPoint.second,
			0, 0, 1);

		// Construct the distortion coefficients
		// k1 k2 p1 p2 k3
		distortionCoefficients = (cv::Mat1d(1, 5) << lensParameters.distortionRadial[0],
			lensParameters.distortionRadial[1],
			lensParameters.distortionTangential.first,
			lensParameters.distortionTangential.second,
			lensParameters.distortionRadial[2]);
	}

private:

	cv::Mat zImage, grayImage;
	cv::Mat cameraMatrix, distortionCoefficients;
	std::mutex flagMutex;


};

int main(int argc, char *argv[])
{

	MyListener listener;

	// this represents the main camera device object
	std::unique_ptr<royale::ICameraDevice> cameraDevice;

	// the camera manager will query for a connected camera
	{
		royale::CameraManager manager;

		// try to open the first connected camera
		royale::Vector<royale::String> camlist(manager.getConnectedCameraList());
		std::cout << "Detected " << camlist.size() << " camera(s)." << std::endl;

		if (!camlist.empty())
		{
			cameraDevice = manager.createCamera(camlist[0]);
		}
		else
		{
			std::cerr << "No suitable camera device detected." << std::endl
				<< "Please make sure that a supported camera is plugged in, all drivers are "
				<< "installed, and you have proper USB permission" << std::endl;
			return 1;
		}

		camlist.clear();

	}
	// the camera device is now available and CameraManager can be deallocated here

	if (cameraDevice == nullptr)
	{
		// no cameraDevice available
		if (argc > 1)
		{
			std::cerr << "Could not open " << argv[1] << std::endl;
			return 1;
		}
		else
		{
			std::cerr << "Cannot create the camera device" << std::endl;
			return 1;
		}
	}

	// call the initialize method before working with the camera device
	auto status = cameraDevice->initialize();
	if (status != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Cannot initialize the camera device, error string : " << getErrorString(status) << std::endl;
		return 1;
	}

	// retrieve the lens parameters from Royale
	royale::LensParameters lensParameters;
	status = cameraDevice->getLensParameters(lensParameters);
	if (status != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Can't read out the lens parameters" << std::endl;
		return 1;
	}

	listener.setLensParameters(lensParameters);

	// register a data listener
	if (cameraDevice->registerDataListener(&listener) != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Error registering data listener" << std::endl;
		return 1;
	}
	cameraDevice->setExposureMode(royale::ExposureMode::AUTOMATIC);
	// start capture mode
	if (cameraDevice->startCapture() != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Error starting the capturing" << std::endl;
		return 1;
	}
	if (Eingabe == "1")//Auswerten
	{
		while (1)
		{
			if (GetAsyncKeyState(32))
				break;
		}
	}

	if (Eingabe == "3")//Lesen!
	{
		uint16_t Framerate;
		cameraDevice->getMaxFrameRate(Framerate);
		listener.Video_Anlegen("Video", listener.zImageColor.rows, listener.zImageColor.cols, Framerate);
		listener.arg = true;
		while (1)
		{
			if (GetAsyncKeyState(32))
				break;
		}
	}

	if (Eingabe == "2")//Schreiben
	{
		std::string Einlesen;
		std::cin >> Einlesen;
		uint16_t Framerate;
		cameraDevice->getMaxFrameRate(Framerate);
		listener.Video_Anlegen(Einlesen, listener.zImageColor.rows, listener.zImageColor.cols, Framerate);
		listener.arg = true;
		while (1)
		{
			if (GetAsyncKeyState(32))
				break;
		}
	}

	// stop capture mode
	if (cameraDevice->stopCapture() != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Error stopping the capturing" << std::endl;
		return 1;
	}

	return 0;
}
