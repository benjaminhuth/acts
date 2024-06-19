// This file is part of the Acts project.
//
// Copyright (C) 2022-2024 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>

#include "TBranch.h"
#include "TTree.h"

namespace ActsExamples {

template<typename T>
class RootReadHandle {
    T *m_value;
    std::string_view m_branchName;

public:
    RootReadHandle(TTree *tree, std::string_view branchName) : m_value(new T()), m_branchName(branchName) {
       tree->SetBranchAddress(m_value, branchName);
    }

    ~RootReadHandle() {
        delete m_value;
    }

    const T& value() const { return *m_value; }
    T& value() { return *m_value; }
};

template<typename T>
class RootWriteHandle {
    T m_value;
    std::string_view m_branchName;

public:
    RootWriteHandle(TTree *tree, std::string_view branchName) {
        tree->Branch(branchName, &m_value);
    }

    const T& value() const { return m_value; }
    T& value() { return m_value; }
};

}
