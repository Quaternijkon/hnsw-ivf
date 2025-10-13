#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

struct BenchmarkConfig {
    struct BuildConfig {
        std::map<std::string, std::vector<int>> params;
        std::vector<std::string> metrics;
    } build;
    
    struct SearchConfig {
        std::map<std::string, std::vector<double>> params;
        std::vector<std::string> metrics;
    } search;
};

class ConfigParser {
public:
    static BenchmarkConfig parseConfig(const std::string& configFile) {
        BenchmarkConfig config;
        std::ifstream file(configFile);
        std::string line;
        std::string current_section = "";
        std::string current_subsection = "";
        
        if (!file.is_open()) {
            throw std::runtime_error("无法打开配置文件: " + configFile);
        }
        
        while (std::getline(file, line)) {
            // 去除前后空白
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t") + 1);
            
            // 跳过空行和注释
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            // 检查是否是section
            if (line == "build" || line == "search") {
                current_section = line;
                current_subsection = "";
                continue;
            }
            
            // 检查是否是subsection
            if (line == "param" || line == "metric") {
                current_subsection = line;
                continue;
            }
            
            // 解析参数或指标
            if (current_subsection == "param") {
                if (current_section == "build") {
                    parseIntParamLine(line, config.build.params);
                } else {
                    parseDoubleParamLine(line, config.search.params);
                }
            } else if (current_subsection == "metric") {
                parseMetricLine(line, current_section == "build" ? config.build.metrics : config.search.metrics);
            }
        }
        
        return config;
    }
    
private:
    static void parseIntParamLine(const std::string& line, std::map<std::string, std::vector<int>>& params) {
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) {
            return;
        }
        
        std::string param_name = line.substr(0, colon_pos);
        param_name.erase(0, param_name.find_first_not_of(" \t"));
        param_name.erase(param_name.find_last_not_of(" \t") + 1);
        
        std::string values_str = line.substr(colon_pos + 1);
        values_str.erase(0, values_str.find_first_not_of(" \t"));
        
        std::vector<int> values;
        std::stringstream ss(values_str);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            values.push_back(std::stoi(value));
        }
        
        params[param_name] = values;
    }
    
    static void parseDoubleParamLine(const std::string& line, std::map<std::string, std::vector<double>>& params) {
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) {
            return;
        }
        
        std::string param_name = line.substr(0, colon_pos);
        param_name.erase(0, param_name.find_first_not_of(" \t"));
        param_name.erase(param_name.find_last_not_of(" \t") + 1);
        
        std::string values_str = line.substr(colon_pos + 1);
        values_str.erase(0, values_str.find_first_not_of(" \t"));
        
        std::vector<double> values;
        std::stringstream ss(values_str);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            values.push_back(std::stod(value));
        }
        
        params[param_name] = values;
    }
    
    static void parseMetricLine(const std::string& line, std::vector<std::string>& metrics) {
        std::stringstream ss(line);
        std::string metric;
        
        while (std::getline(ss, metric, ',')) {
            metric.erase(0, metric.find_first_not_of(" \t"));
            metric.erase(metric.find_last_not_of(" \t") + 1);
            if (!metric.empty()) {
                metrics.push_back(metric);
            }
        }
    }
};

#endif // CONFIG_PARSER_H
