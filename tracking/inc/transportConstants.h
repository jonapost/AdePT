// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr float kPushLinear = 1.0e-8; // * copcore::units::millimeter;

enum ETransportCode {
   kBoundaryLinear        = -1,  //  At boundary after linear navigation
   kBoundaryWithField     = -2,  //  At boundary after field propagation
   kUnfinishedIntegration = -3,  //  'In flight' - not yet at boundary, integration must continue
};
