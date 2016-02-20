#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <numeric>
#include <iostream>
#include <stdio.h>
#ifdef WIN32
#include <Windows.h>
#define WIN32_LEAN_AND_MEAN
#endif

using namespace std;
using namespace cv;


int
chunkSize = 20,
frames_n = 2,
magnitude_threshold = 3,
angle_threshold = 45,
total_threshold = 10,
average_space = 0, // Number of pixels above/below/left/right of working pixel to average over
time_threshold = 0,
size_min = 10000,
smoothing_size = 0;

const string
trackbarWindowName("Trackbars"),
videoWindowName("video"),
x_componentWindowName("X velocity"),
x_thresholdedWindowName("X velocity thresholded");

void remove_contours(vector<vector<Point>>& contours, double cmin, double cmax)
{
	auto it = partition(begin(contours), end(contours), [=](vector<Point> const& data)
	{
		double const size = contourArea(data);
		return cmin <= size && size <= cmax;
	});
	contours.erase(it, end(contours));
}

void getFrameDerivatives(Mat& input, Mat& partial_x, Mat& partial_y)
{
	const double partial_array[] = { 0.5, -0.5 };

	Mat	partial_x_matrix(Size(2, 1), CV_64F, (void*)partial_array);
	Mat	partial_y_matrix(Size(1, 2), CV_64F, (void*)partial_array);

	filter2D(input, partial_x, -1, partial_x_matrix);
	filter2D(input, partial_y, -1, partial_y_matrix);
}

void getDerivatives(deque<Mat>& frames, Mat& partialX, Mat& partialY, Mat& partialT)
{
	Mat
		partialX_show,
		partialY_show,
		partialT_show;
	vector<Mat>
		partials_x,
		partials_y;

	const Mat zeros = Mat::zeros(frames[0].size(), frames[0].type());
	partialX = zeros.clone();
	partialY = zeros.clone();

	deque<Mat> differences(frames.size());
	adjacent_difference(frames.begin(), frames.end(), differences.begin());
	differences[0] = zeros;
	partialT = accumulate(differences.begin(), differences.end(), zeros) / double(frames.size());
	for (Mat& frame : frames)
	{
		Mat partial_x_temp, partial_y_temp;
		getFrameDerivatives(frame, partial_x_temp, partial_y_temp);
		partials_x.push_back(partial_x_temp);
		partials_y.push_back(partial_y_temp);
	}
	for (Mat& partial_x : partials_x)
	{
		copyMakeBorder(partial_x, partial_x, average_space / 2, average_space - average_space / 2, 0, 0, BORDER_REPLICATE);
		for (int i(0); i <= average_space; ++i)
			partialX += partial_x.rowRange(i, partial_x.rows - average_space + i);
	}
	for (Mat& partial_y : partials_y)
	{
		copyMakeBorder(partial_y, partial_y, 0, 0, average_space / 2, average_space - average_space / 2, BORDER_REPLICATE);
		for (int i(0); i <= average_space; ++i)
			partialY += partial_y.colRange(i, partial_y.cols - average_space + i);
	}
	partialX /= double(frames.size() * (average_space + 1));
	partialY /= double(frames.size() * (average_space + 1));
	normalize(partialX, partialX_show, 0, 1, NORM_MINMAX);
	normalize(partialY, partialY_show, 0, 1, NORM_MINMAX);
	normalize(partialT, partialT_show, 0, 1, NORM_MINMAX);
	/*
	imshow("px", partialX_show);
	imshow("py", partialY_show);*/
	//imshow("pt", partialT_show);
	//
}

pair<vector<double>, vector<double>> LKTracker(const vector<Point2i> positions, const int size, const Mat_<double>& partialX, const Mat_<double>& partialY, const Mat_<double>& partialT)
{
	pair<vector<double>, vector<double>> velocity;
	vector<double>& x_velocity = velocity.first;
	vector<double>& y_velocity = velocity.second;

	for (Point2i position : positions)
	{
		Matx<double, 2, 2> A(0, 0, 0, 0);
		Vec2d B(0, 0);
		const int
			y_min = max(0, position.y - size),
			y_max = min(position.y + size, partialX.rows),
			x_min = max(0, position.x - size),
			x_max = min(position.x + size, partialX.cols);
		for (int y(y_min); y < y_max; ++y)
			for (int x(x_min); x < x_max; ++x)
			{
			A(0, 0) += partialX[y][x] * partialX[y][x];
			A(0, 1) += partialX[y][x] * partialY[y][x];
			A(1, 1) += partialY[y][x] * partialY[y][x];
			B[0] += partialT[y][x] * partialX[y][x];
			B[1] += partialT[y][x] * partialY[y][x];
			}
		A(1, 0) = A(0, 1);
		Vec2d temp = A.inv()*B;
		x_velocity.push_back(temp[0]);
		y_velocity.push_back(temp[1]);
	}

	return velocity;
}

int main(int argc, const char** argv)
{
	VideoCapture cap;
	if (argc > 1)
		cap.open(string(argv[1]));
	else
		cap.open(CV_CAP_ANY);

	if (!cap.isOpened())
		printf("Error: could not load a camera or video.\n");

#ifdef WIN32
	SetConsoleTitle(L"Average X velocity");
	MoveWindow(GetConsoleWindow(), 0, 0, 300, 1000, false);
#endif

	namedWindow(videoWindowName);
	namedWindow(x_componentWindowName);
	namedWindow(x_thresholdedWindowName);
	namedWindow(trackbarWindowName);
	moveWindow(videoWindowName, 300, 0);
	moveWindow(x_componentWindowName, 1200, 0);
	moveWindow(x_thresholdedWindowName, 300, 520);
	moveWindow(trackbarWindowName, 1200, 520);

	// Create trackbars for various constants
	createTrackbar("chunkSize", trackbarWindowName, &chunkSize, 100);
	createTrackbar("frames_n", trackbarWindowName, &frames_n, 8);
	createTrackbar("magnitude", trackbarWindowName, &magnitude_threshold, 100);
	createTrackbar("angle", trackbarWindowName, &angle_threshold, 90);
	createTrackbar("total", trackbarWindowName, &total_threshold, 60);
	createTrackbar("average_space", trackbarWindowName, &average_space, 20);
	createTrackbar("time_threshold", trackbarWindowName, &time_threshold, 10);
	createTrackbar("size_min", trackbarWindowName, &size_min, 15000);
	createTrackbar("smoothing_size", trackbarWindowName, &smoothing_size, 5);

	resizeWindow(trackbarWindowName, 300, 120);


	Mat original;
	deque<Mat> frames; // A circular queue of frames used in processing
	Mat frame;
	Mat partialX, partialY, partialT;
	vector<Point2i> positions;

	// Get video dimensions
	cap >> original;

	// Generate grid of blocks to LK over
	int lastChunkSize = 0;
	for (;;)
	{
		// 50 frames per second
		waitKey(20);

		// Stream video input
		cap >> original;
		if (!original.data)
		{
			printf("Error: no frame data.\n");
			break;
		}

		if (chunkSize != lastChunkSize)
		{
			if (!chunkSize)
				chunkSize = 1;
			lastChunkSize = chunkSize;
			positions.clear();
			for (int y = chunkSize / 2; y < original.rows - chunkSize / 2; y += chunkSize)
				for (int x = chunkSize / 2; x < original.cols - chunkSize / 2; x += chunkSize)
					positions.push_back(Point(x, y));
		}

		// The images of the x components of the LK with zero as neutral grey
		Mat_<double> x_component(original.rows, original.cols, 0.5);
		Mat_<double> x_thresholded(original.rows, original.cols, 0.5);
		Mat_<uchar> left_thresholded(original.rows, original.cols, uchar());
		Mat_<uchar> right_thresholded(original.rows, original.cols, uchar());
		Mat_<uchar> up_thresholded(original.rows, original.cols, uchar());
		Mat_<uchar> down_thresholded(original.rows, original.cols, uchar());

		// Flip the video so that the webcam is like a mirror
		flip(original, frame, 1);
		original = frame;

		// Convert to double representation to avoid range issues
		cvtColor(frame, frame, CV_BGR2GRAY);
		frame.convertTo(frame, CV_64F);

		// Wait until enough frames have been received before processing
		if (frames.size() < frames_n)
		{
			frames.push_front(frame.clone());
			continue;
		}
		if (frames.size() > frames_n)
			frames.erase(frames.begin() + frames_n, frames.end());
		frames.pop_back();
		frames.push_front(frame.clone());

		// Calculate spatial and temporal partial derivatives for all pixels
		getDerivatives(frames, partialX, partialY, partialT);

		// Perform LK to estimate velocities of each block
		pair<vector<double>, vector<double>> velocity = LKTracker(positions, chunkSize, partialX, partialY, partialT);
		vector<double>& x_velocity = velocity.first;
		vector<double>& y_velocity = velocity.second;

		// Perfom a max-max normalisation
		vector<double> x_velocity_normalised = x_velocity;
		static double
			x_velocity_global_maximum(0.0),
			x_velocity_global_minimum(0.0);
		double
			x_velocity_local_maximum(*max_element(x_velocity_normalised.begin(), x_velocity_normalised.end())),
			x_velocity_local_minimum(*min_element(x_velocity_normalised.begin(), x_velocity_normalised.end()));
		x_velocity_global_maximum = max(x_velocity_global_maximum, x_velocity_local_maximum);
		x_velocity_global_minimum = min(x_velocity_global_minimum, x_velocity_local_minimum);
		for (double& v : x_velocity_normalised)
			v = (v - x_velocity_global_minimum) / (x_velocity_global_maximum - x_velocity_global_minimum);

		// Old per-frame normalisation:
		// normalize(x_velocity_normalised, x_velocity_normalised, 0, 1, NORM_MINMAX);
		/*
		// Shows most extreme velocity components in current frame
		cout
		<< "y max " << *std::max_element(y_velocity.begin(), y_velocity.end()) << ",\t y min " << *min_element(y_velocity.begin(), y_velocity.end()) << ",\t "
		<< "x max " << *std::max_element(x_velocity.begin(), x_velocity.end()) << ",\t x min " << *min_element(x_velocity.begin(), x_velocity.end()) << endl;
		//*/

		// Calculate average velocity of blocks subject to thesholding
		double total_x = 0.0;
		double total_y = 0.0;
		int total_cells_x = 0;
		int total_cells_y = 0;
#pragma omp parallel for reduction(+: total_x, total_y, total_cells_x, total_cells_y)
		for (int i = 0; i < x_velocity.size(); ++i)
		{
			// Draw raw x velocity for each block
			rectangle(x_component, Rect(positions[i].x - chunkSize / 2, positions[i].y - chunkSize / 2, chunkSize, chunkSize), Scalar(x_velocity_normalised[i]), CV_FILLED);

			// Magnitude threshold
			if (hypot(x_velocity[i], y_velocity[i]) < magnitude_threshold)
				continue;

			// Draw velocity vector
			line(original, positions[i], positions[i] + Point(int(x_velocity[i]), int(y_velocity[i])), Scalar(255), 2);
			circle(original, positions[i], 1, Scalar(0, 255, 0), 1);

			// Angle threshold
			if (fastAtan2(float(abs(y_velocity[i])), float(abs(x_velocity[i]))) < angle_threshold)
			{

				// Add to total x velocity
				total_x += x_velocity[i];
				++total_cells_x;

				// Draw binary thresholded velocity
				rectangle(x_thresholded, Rect(positions[i].x - chunkSize / 2, positions[i].y - chunkSize / 2, chunkSize, chunkSize), Scalar(int(x_velocity[i] > 0)), CV_FILLED);

				// yes
				if (velocity.first[i] > 0)
					rectangle(left_thresholded, Rect(positions[i].x - chunkSize / 2, positions[i].y - chunkSize / 2, chunkSize, chunkSize), Scalar(255), CV_FILLED);
				else
					rectangle(right_thresholded, Rect(positions[i].x - chunkSize / 2, positions[i].y - chunkSize / 2, chunkSize, chunkSize), Scalar(255), CV_FILLED);
			}
			if (fastAtan2(float(abs(x_velocity[i])), float(abs(y_velocity[i]))) < angle_threshold)
			{
				// Add to total x velocity
				total_y += y_velocity[i];
				++total_cells_y;

				// yes as well
				if (velocity.second[i] > 0)
					rectangle(up_thresholded, Rect(positions[i].x - chunkSize / 2, positions[i].y - chunkSize / 2, chunkSize, chunkSize), Scalar(255), CV_FILLED);
				else
					rectangle(down_thresholded, Rect(positions[i].x - chunkSize / 2, positions[i].y - chunkSize / 2, chunkSize, chunkSize), Scalar(255), CV_FILLED);

			}
		}

		// Find contours

		if (smoothing_size)
		{
			smoothing_size *= 2 - 1;
			blur(left_thresholded, left_thresholded, Size(chunkSize*smoothing_size, chunkSize*smoothing_size));
			blur(right_thresholded, right_thresholded, Size(chunkSize*smoothing_size, chunkSize*smoothing_size));
			blur(up_thresholded, up_thresholded, Size(chunkSize*smoothing_size, chunkSize*smoothing_size));
			blur(down_thresholded, down_thresholded, Size(chunkSize*smoothing_size, chunkSize*smoothing_size));
		}

		vector<vector<Point>> left_contours;
		vector<vector<Point>> right_contours;
		vector<vector<Point>> up_contours;
		vector<vector<Point>> down_contours;

		findContours(left_thresholded, left_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		remove_contours(left_contours, size_min, 500000);
		for (int i = 0; i< left_contours.size(); i++)
		{
			Scalar color = Scalar(0, 0, 255);
			drawContours(original, left_contours, i, color, 5, 8);
		}

		findContours(right_thresholded, right_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		remove_contours(right_contours, size_min, 500000);
		for (int i = 0; i< right_contours.size(); i++)
		{
			Scalar color = Scalar(0, 255, 0);
			drawContours(original, right_contours, i, color, 5, 8);
		}

		findContours(up_thresholded, up_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		remove_contours(up_contours, size_min, 500000);
		for (int i = 0; i< up_contours.size(); i++)
		{
			Scalar color = Scalar(0, 255, 255);
			drawContours(original, up_contours, i, color, 5, 8);
		}

		findContours(down_thresholded, down_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		remove_contours(down_contours, size_min, 500000);
		for (int i = 0; i< down_contours.size(); i++)
		{
			Scalar color = Scalar(255, 255, 0);
			drawContours(original, down_contours, i, color, 5, 8);
		}

		// Display detected gesture
		static int time(0);
		static enum { left, right, none } status(none);
		if (!total_cells_x)
		{
			status = none;
			time = 0;
		}
		else
		{
			// Write average velocity to console
			total_x /= double(total_cells_x);
			cout << int(total_x) << endl;

			if (total_x > total_threshold)
			{
				if (status == left)
					++time;
				else if (status == right)
					time = 0;
				status = left;
			}
			else if (total_x < -total_threshold)
			{
				if (status == right)
					++time;
				else if (status == left)
					time = 0;
				status = right;
			}
			else
			{
				status = none;
				// time = 0;
			}
		}
		string status_string;
		if (status == none)
			status_string = "none";
		else if (time < time_threshold)
			status_string = "none - " + to_string(time_threshold - time);
		else if (status == left)
			status_string = "left";
		else
			status_string = "right";
		putText(x_thresholded, status_string, Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(1));

		imshow(videoWindowName, original);
		imshow(x_componentWindowName, x_component);
		imshow(x_thresholdedWindowName, x_thresholded);
	}
}