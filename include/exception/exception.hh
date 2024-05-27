#ifndef B8BA91A7_978F_4918_AF25_A486CE4660C0
#define B8BA91A7_978F_4918_AF25_A486CE4660C0

#include <sstream>
#include <stdexcept>

namespace psz {

struct exception_placeholder : public std::exception {
  std::string err_msg;
  exception_placeholder()
  {
    std::stringstream ss;
    ss << __FILE__ << ':' << __LINE__ << ": not implemented\n";
    err_msg = ss.str();
  }
  const char* what() const noexcept { return err_msg.c_str(); }
};

struct exception_incorrect_type : public std::exception {
  std::string err_msg;
  exception_incorrect_type(const char* buf_name)
  {
    std::stringstream ss;
    ss << buf_name << ": incorrect datatype.";
    err_msg = ss.str();
  }
  const char* what() const noexcept { return err_msg.c_str(); }
};

}  // namespace psz

#define NONEXIT_CATCH(cpp_execept, ENUM_PSZERR) \
  catch (const cpp_execept& e)                  \
  {                                             \
    std::cerr << e.what() << std::endl;         \
    return ENUM_PSZERR;                         \
  }

#endif /* B8BA91A7_978F_4918_AF25_A486CE4660C0 */
