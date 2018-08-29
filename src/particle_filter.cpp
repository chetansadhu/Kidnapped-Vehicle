/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <map>

#include "particle_filter.h"


using namespace std;

#define NUMBER_OF_PARTICLES 50
#define YAW_EPSILON 0.001f
#define SEP ", "


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);
	std::default_random_engine gen;
	num_particles = NUMBER_OF_PARTICLES;
	for (int i = 0; i < num_particles; ++i) {
		weights.push_back(1.0);
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
	}
	is_initialized = true;
	log = new Logger("ParticleFilter.log");
	log->LOG(INFO, __FUNCTION__, "Particles are intialized");

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double xf = 0.0, yf = 0.0, thetaf = 0.0;
	const double mean_noise = 0.0;
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(mean_noise, std_pos[0]);
	std::normal_distribution<double> dist_y(mean_noise, std_pos[1]);
	std::normal_distribution<double> dist_theta(mean_noise, std_pos[2]);
#ifdef _DEBUG
	std::ostringstream str;
#endif
	for (int i = 0; i < num_particles; ++i) {
#ifdef _DEBUG
		str << "\n" << i << SEP << particles[i].x << SEP << particles[i].y << SEP << particles[i].theta;
#endif
		if (fabs(yaw_rate) > YAW_EPSILON) {
			xf = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			yf = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
		}
		else {
			xf = particles[i].x + velocity * cos(particles[i].theta) * delta_t;
			yf = particles[i].y + velocity * sin(particles[i].theta) * delta_t;
		}
		thetaf = particles[i].theta + yaw_rate * delta_t;

		double noise_x = dist_x(gen);
		double noise_y = dist_y(gen);
		double noise_theta = dist_theta(gen);

		particles[i].x = xf + noise_x;
		particles[i].y = yf + noise_y;
		particles[i].theta = thetaf + noise_theta;

#ifdef _DEBUG
		str << SEP << particles[i].x << SEP << particles[i].y << SEP << particles[i].theta;
		str << SEP << velocity << SEP << yaw_rate << SEP << delta_t;
		str << SEP << noise_x << SEP << noise_y << SEP << noise_theta;
#endif

	}
#ifdef _DEBUG
	log->LOG(DEBUG, "Predicted measurement", str.str().c_str());
#endif
	log->LOG(INFO, __FUNCTION__, "Predicted Particles");
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
#ifdef _DEBUG
	std::ostringstream str1;
	char msg[1024];
	sprintf(msg, "Observations size: %d, predicted measurement size: %d", observations.size(), predicted.size());
	log->LOG(DEBUG, __FUNCTION__, msg);
#endif
	for (unsigned int i = 0; i < observations.size(); ++i) {
		double min_val = std::numeric_limits<double>::max();
		double xo = observations[i].x;
		double yo = observations[i].y;
//		int debug_id;
		for (unsigned int j = 0; j < predicted.size(); ++j) {
			double xp = predicted[j].x;
			double yp = predicted[j].y;
			double distance = dist(xp, yp, xo, yo);
			if (distance < min_val) {
				min_val = distance;
				observations[i].id = predicted[j].id;
//				debug_id = j;
			}
#ifdef _DEBUG
			str1 << "\n";
			str1 << xo << SEP << yo << SEP;
			str1 << xp << SEP << yp << SEP;
			str1 << distance << SEP << predicted[j].id;
#endif
		}
	}
#ifdef _DEBUG
	log->LOG(DEBUG, "DataAssociation: ", str1.str().c_str());
#endif

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double sum_weights = 0.0;
	for (int i = 0; i < num_particles; ++i) {
		weights[i] = 1.0;
		double xp = particles[i].x;
		double yp = particles[i].y;
		double theta = particles[i].theta;
#ifdef _DEBUG
		std::ostringstream str2;
		str2 << "\n" << xp << SEP << yp << SEP << theta;
		log->LOG(DEBUG, "Particle details", str2.str().c_str());
		str2.str("");
#endif
		std::vector<LandmarkObs> transformed_observations;
		for (unsigned int j = 0; j < observations.size(); ++j) {
			double xc = observations[j].x;
			double yc = observations[j].y;
			int id = observations[i].id;

			double xm = 0.0, ym = 0.0;
			LandmarkObs tobs;
			Transform_Car_Map(xp, yp, xc, yc, theta, xm, ym);
			tobs.x = xm;
			tobs.y = ym;
			tobs.id = id;
			transformed_observations.push_back(tobs);
#ifdef _DEBUG
			str2 << "\n" << xc << SEP << yc;
			str2 << SEP << xm << SEP << ym;
#endif
		}

#ifdef _DEBUG
		log->LOG(DEBUG, "Transformed observation", str2.str().c_str());
		str2.str("");
#endif
		std::vector <LandmarkObs> predicted;
		std::map<int, int> index_map;
		for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
			double xlm = map_landmarks.landmark_list[k].x_f;
			double ylm = map_landmarks.landmark_list[k].y_f;
			int id_lm = map_landmarks.landmark_list[k].id_i;
			index_map[id_lm] = -1;
			double distance = dist(xp, yp, xlm, ylm);
			if (distance <= sensor_range) {
				LandmarkObs pred;
				pred.x = xlm;
				pred.y = ylm;
				pred.id = id_lm;
				index_map[pred.id] = k;
				predicted.push_back(pred);
			}
#ifdef _DEBUG
			str2 << "\n" << xlm << SEP << ylm << SEP << distance;
#endif
		}
#ifdef _DEBUG
		char msg[1024];
		sprintf(msg, "Landmark list size: %d, Landmarks within particle sensor range: %d", map_landmarks.landmark_list.size(), predicted.size());
		log->LOG(DEBUG, msg, str2.str().c_str());
		str2.str("");
#endif
		dataAssociation(predicted, transformed_observations);
		double weight = 1.0;

		for (unsigned int j = 0; j < transformed_observations.size(); ++j) {
			double x_obs = transformed_observations[j].x;
			double y_obs = transformed_observations[j].y;
			int lId = index_map[transformed_observations[j].id];
			double mu_x = 0.0;
			double mu_y = 0.0;
			if (lId >= 0) {
				mu_x = map_landmarks.landmark_list[lId].x_f;
				mu_y = map_landmarks.landmark_list[lId].y_f;
			}
			if (std_landmark[0] != 0 && std_landmark[1] != 0) {
				double exponent = ((x_obs - mu_x) * (x_obs - mu_x))/(2 * std_landmark[0] * std_landmark[0]) +
								  ((y_obs - mu_y) * (y_obs - mu_y))/(2 * std_landmark[1] * std_landmark[1]);
				double gauss_norm = 1.0 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
				weight = exp(-1.0 * exponent) * gauss_norm;
				weights[i] *= weight;
#ifdef _DEBUG
				str2 << "\n";
				str2 << x_obs << SEP << y_obs << SEP << mu_x << SEP << mu_y << SEP << exponent << SEP << gauss_norm;
				str2 << SEP << weight;
#endif
			}
		}
		particles[i].weight = weights[i];
		sum_weights += weights[i];
#ifdef _DEBUG
		log->LOG(DEBUG, "Updated weight calculation", str2.str().c_str());
		str2.str("");
		char val[1024];
		sprintf(val, "Particle final weight %lf\n", particles[i].weight);
		log->LOG(DEBUG, __FUNCTION__, val);
#endif

	}
	std::ostringstream str1;
	for (int i = 0; i < num_particles; ++i) {
		if (sum_weights != 0) {
			weights[i] /= sum_weights;
		}
		else {
			weights[i] = 0.0;
		}
		str1 << SEP << weights[i];
	}
	log->LOG(INFO, "Normalized weights\n", str1.str().c_str());
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<Particle> resampled_particles(num_particles);
	std::discrete_distribution<int> dist_weights(weights.begin(), weights.end());
	std::default_random_engine gen;
	std::ostringstream str1;
	for (int i = 0; i < num_particles; ++i) {
		int index = dist_weights(gen);
		resampled_particles[i] = particles[index];
		str1 << SEP << index;
	}
	particles = resampled_particles;
	log->LOG(INFO, "Resampled particle index\n", str1.str().c_str());
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

void ParticleFilter::Transform_Car_Map(const double xp, const double yp,
		const double xc, const double yc, const double theta,
		double &xm, double &ym) {
	xm = xp + (cos(theta) * xc) - (sin(theta) * yc);
	ym = yp + (sin(theta) * xc) + (cos(theta) * yc);
}
