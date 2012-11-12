#ifndef LEX_CAST_H_HFAKJSHNKJSNHKVSNHDJG
#define LEX_CAST_H_HFAKJSHNKJSNHKVSNHDJG

#include <sstream>
#include <typeinfo>

template<typename Target, typename Source>
  Target lexical_cast(Source arg)
  {
    std::stringstream interpreter;
    Target result;

    if (!(interpreter << arg) || !(interpreter >> result) || !(interpreter >> std::ws).eof())
      throw std::bad_cast();
    return result;
  }

#endif /* LEX_CAST_H_HFAKJSHNKJSNHKVSNHDJG */

