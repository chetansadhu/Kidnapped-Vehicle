/*
 * Logger.h
 *
 *  Created on: Jul 31, 2018
 *      Author: chetan
 */

#ifndef LOGGER_H_
#define LOGGER_H_

#include <fstream>
#include <time.h>
#include <string>

enum typelog {
	DEBUG,
	INFO,
	WARN,
	ERROR
};

class Logger {
private:
	std::ofstream log_file_;

private:
	const std::string CurrentDateTime() {
		time_t t = time(0);
		struct tm *now = localtime(&t);
		char buf[255];
		strftime(buf, sizeof(buf), "%d-%b-%Y %H:%M:%S", now);
		return buf;
	}

public:
	Logger(const char *name) {
#ifdef LOGGING
		if (name != NULL) {
			log_file_.open(name, std::ios_base::app);
			if (log_file_.is_open()) {
				log_file_ << "\n------------------------------------------------------------------------------------\n";
				log_file_ << CurrentDateTime() << "\n";
			}
			else {
				throw;
			}
		}
		else {
			throw;
		}
#endif
	}
	~Logger() {
#ifdef LOGGING
		if (log_file_.is_open()) {
			log_file_.close();
		}
#endif
	}

	void LOG(typelog t, const char* function, const char *msg) {
#ifdef LOGGING
		if (!log_file_.is_open()) return;
		switch (t) {
		case DEBUG:
			log_file_ << "DEBUG: " << function << ": " << msg << std::endl;
			break;

		case INFO:
			log_file_ << "INFO: " << function << ": " << msg << std::endl;
			break;

		case WARN:
			log_file_ << "WARN: " << function << ": " << msg << std::endl;
			break;

		case ERROR:
			log_file_ << "ERROR: " << function << ": " << msg << std::endl;
			break;

		default:
			break;
		}
#endif
	}
};



#endif /* LOGGER_H_ */
