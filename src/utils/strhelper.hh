/**
 * @file strhelper.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-19
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_UTILS_STRHELPER_HH
#define CUSZ_UTILS_STRHELPER_HH

#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "format.hh"

using std::cerr;
using std::endl;

using ss_t     = std::stringstream;
using map_t    = std::unordered_map<std::string, std::string>;
using str_list = std::vector<std::string>;

struct StrHelper {
    static unsigned int str2int(std::string s)
    {
        char* end;
        auto  res = std::strtol(s.c_str(), &end, 10);
        if (*end) {
            const char* notif = "invalid option value, non-convertible part: ";
            cerr << LOG_ERR << notif << "\e[1m" << s << "\e[0m" << endl;
        }
        return res;
    };

    static unsigned int str2int(const char* s)
    {
        char* end;
        auto  res = std::strtol(s, &end, 10);
        if (*end) {
            const char* notif = "invalid option value, non-convertible part: ";
            cerr << LOG_ERR << notif << "\e[1m" << s << "\e[0m" << endl;
        }
        return res;
    };

    static double str2fp(std::string s)
    {
        char* end;
        auto  res = std::strtod(s.c_str(), &end);
        if (*end) {
            const char* notif = "invalid option value, non-convertible part: ";
            cerr << LOG_ERR << notif << "\e[1m" << end << "\e[0m" << endl;
        }
        return res;
    }

    static double str2fp(const char* s)
    {
        char* end;
        auto  res = std::strtod(s, &end);
        if (*end) {
            const char* notif = "invalid option value, non-convertible part: ";
            cerr << LOG_ERR << notif << "\e[1m" << end << "\e[0m" << endl;
        }
        return res;
    };

    static bool is_kv_pair(std::string s) { return s.find("=") != std::string::npos; }

    static std::pair<std::string, std::string> separate_kv(std::string& s)
    {
        std::string delimiter = "=";

        if (s.find(delimiter) == std::string::npos)
            throw std::runtime_error("\e[1mnot a correct key-value syntax, must be \"opt=value\"\e[0m");

        std::string k = s.substr(0, s.find(delimiter));
        std::string v = s.substr(s.find(delimiter) + delimiter.length(), std::string::npos);

        return std::make_pair(k, v);
    }

    static void parse_strlist_as_kv(const char* in_str, map_t& kv_list)
    {
        ss_t ss(in_str);
        while (ss.good()) {
            std::string tmp;
            std::getline(ss, tmp, ',');
            kv_list.insert(separate_kv(tmp));
        }
    }

    static void parse_strlist(const char* in_str, str_list& list)
    {
        ss_t ss(in_str);
        while (ss.good()) {
            std::string tmp;
            std::getline(ss, tmp, ',');
            list.push_back(tmp);
        }
    }

    static std::pair<std::string, bool> parse_kv_onoff(std::string in_str)
    {
        auto       kv_literal = "(.*?)=(on|ON|off|OFF)";
        std::regex kv_pattern(kv_literal);
        std::regex onoff_pattern("on|ON|off|OFF");

        bool        onoff = false;
        std::string k, v;

        std::smatch kv_match;
        if (std::regex_match(in_str, kv_match, kv_pattern)) {
            // the 1st match: whole string
            // the 2nd: k, the 3rd: v
            if (kv_match.size() == 3) {
                k = kv_match[1].str();
                v = kv_match[2].str();

                std::smatch v_match;
                if (std::regex_match(v, v_match, onoff_pattern)) {  //
                    onoff = (v == "on") or (v == "ON");
                }
                else {
                    throw std::runtime_error("not (k=v)-syntax");
                }
            }
        }
        return std::make_pair(k, onoff);
    }

    static string doc_format(const string& s)
    {
        std::regex  gray("%(.*?)%");
        std::string gray_text("\e[37m$1\e[0m");

        std::regex  bful("@(.*?)@");
        std::string bful_text("\e[1m\e[4m$1\e[0m");
        std::regex  bf("\\*(.*?)\\*");
        std::string bf_text("\e[1m$1\e[0m");
        std::regex  ul(R"(_((\w|-|\d|\.)+?)_)");
        std::string ul_text("\e[4m$1\e[0m");
        std::regex  red(R"(\^\^(.*?)\^\^)");
        std::string red_text("\e[31m$1\e[0m");

        auto a = std::regex_replace(s, bful, bful_text);
        auto b = std::regex_replace(a, bf, bf_text);
        auto c = std::regex_replace(b, ul, ul_text);
        auto d = std::regex_replace(c, red, red_text);
        auto e = std::regex_replace(d, gray, gray_text);

        return e;
    }
};

#endif