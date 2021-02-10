// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <string>
#include <vector>
#include <istream>
#include <sstream>

class CSVRow
{
    std::string m_line;
    std::vector<std::size_t> m_data;
    
public:
    auto operator[](std::size_t index) const
    {        
        return std::string_view(&m_line[m_data[index] + 1ul], m_data[index + 1ul] -  (m_data[index] + 1ul));
    }
    
    auto size() const
    {
        return m_data.size() - 1ul;
    }
    
    friend inline std::istream& operator>>(std::istream& str, CSVRow& data)
    {
        data.readNextRow(str);
        return str;
    } 
    
    const auto &full_string() const
    {
        return m_line;
    }
    
private:
    void readNextRow(std::istream& str)
    {
        std::getline(str, m_line);

        m_data.clear();
        m_data.emplace_back(-1);
        std::string::size_type pos = 0;
        while((pos = m_line.find(',', pos)) != std::string::npos)
        {
            m_data.emplace_back(pos);
            ++pos;
        }
        
        // This checks for a trailing comma with no data after it.
        pos   = m_line.size();
        m_data.emplace_back(pos);
    }
};



class CSVIterator
{   
    std::istream*       m_str;
    CSVRow              m_row;
    
public:
    using iterator_category = std::input_iterator_tag;
    using value_type = CSVRow;
    using difference_type = std::size_t;
    using pointer = CSVRow*;
    using reference = CSVRow&;

    CSVIterator(std::istream& str)  :m_str(str.good() ? &str : nullptr) { ++(*this); }
    CSVIterator()                   :m_str(nullptr) {}

    CSVIterator& operator++()               {if (m_str) { if (!((*m_str) >> m_row)){m_str = nullptr;}} return *this;}
    CSVIterator operator++(int)             {CSVIterator    tmp(*this);++(*this); return tmp;}
    CSVRow const& operator*()   const       {return m_row;}
    CSVRow const* operator->()  const       {return &m_row;}

    bool operator==(CSVIterator const& rhs) const {return ((this == &rhs) || ((this->m_str == nullptr) && (rhs.m_str == nullptr)));}
    bool operator!=(CSVIterator const& rhs) const {return !((*this) == rhs);}
};


class CSVRange
{
    std::istream&   stream;
    
    public:
        CSVRange(std::istream& str) : stream(str) {}
        CSVIterator begin() const {return CSVIterator{stream};}
        CSVIterator end()   const {return CSVIterator{};}
};
