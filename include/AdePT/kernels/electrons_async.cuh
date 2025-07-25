// // SPDX-FileCopyrightText: 2022 CERN
// // SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/AsyncAdePTTransportStruct.cuh>

#include <AdePT/navigation/BVHNavigator.h>
#include <AdePT/magneticfield/fieldPropagatorConstBz.h>

#include <AdePT/copcore/PhysicalConstants.h>
#include <AdePT/core/AdePTScoringTemplate.cuh>

#include <G4HepEmElectronManager.hh>
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmElectronInteractionBrem.hh>
#include <G4HepEmElectronInteractionIoni.hh>
#include <G4HepEmElectronInteractionUMSC.hh>
#include <G4HepEmPositronInteractionAnnihilation.hh>
// Pull in implementation.
#include <G4HepEmRunUtils.icc>
#include <G4HepEmInteractionUtils.icc>
#include <G4HepEmElectronManager.icc>
#include <G4HepEmElectronInteractionBrem.icc>
#include <G4HepEmElectronInteractionIoni.icc>
#include <G4HepEmElectronInteractionUMSC.icc>
#include <G4HepEmPositronInteractionAnnihilation.icc>

namespace {

// Compute velocity based on the kinetic energy of the particle
__device__ double GetVelocity(double eKin)
{
  // Taken from G4DynamicParticle::ComputeBeta
  const double T    = eKin / copcore::units::kElectronMassC2;
  const double beta = sqrt(T * (T + 2.)) / (T + 1.0);
  return copcore::units::kCLight * beta;
}

} // namespace

namespace AsyncAdePT {

// Compute the physics and geometry step limit, transport the electrons while
// applying the continuous effects and maybe a discrete process that could
// generate secondaries.
template <bool IsElectron, typename Scoring>
static __device__ __forceinline__ void TransportElectrons(Track *electrons, const adept::MParray *active,
                                                          Secondaries &secondaries, adept::MParray *activeQueue,
                                                          adept::MParray *leakedQueue, Scoring *userScoring)
{
  using VolAuxData = adeptint::VolAuxData;
#ifdef VECGEOM_FLOAT_PRECISION
  const Precision kPush = 10 * vecgeom::kTolerance;
#else
  const Precision kPush = 0.;
#endif
  constexpr Precision kPushOutRegion = 10 * vecgeom::kTolerance;
  constexpr int Charge               = IsElectron ? -1 : 1;
  constexpr double restMass          = copcore::units::kElectronMassC2;
  constexpr unsigned short maxSteps  = 10000; // 20'000;
  fieldPropagatorConstBz fieldPropagatorBz(BzFieldValue);

  const int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = electrons[slot];
    currentTrack.stepCounter++; // step counter is already increased below when checking for maxSteps
    auto eKin                = currentTrack.eKin;
    const auto preStepEnergy = eKin;
    auto pos                 = currentTrack.pos;
    const auto preStepPos{pos};
    auto dir = currentTrack.dir;
    vecgeom::Vector3D<Precision> preStepDir(dir);
    auto navState     = currentTrack.navState;
    const auto volume = navState.Top();
    // the MCC vector is indexed by the logical volume id
    const int lvolID = volume->GetLogicalVolume()->id();
    // TODO: Why do we have a specific AsyncAdePT VolAuxData?
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID];

    assert(auxData.fGPUregion > 0); // make sure we don't get inconsistent region here
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    auto survive = [&](bool leak = false) {
      currentTrack.eKin     = eKin;
      currentTrack.pos      = pos;
      currentTrack.dir      = dir;
      currentTrack.navState = navState;
      if (leak)
        leakedQueue->push_back(slot);
      else
        activeQueue->push_back(slot);
    };

    if (auxData.fGPUregion == 0) {
      printf(__FILE__ ":%d Error: Should kick particle %d lvol %d (%15.12f %15.12f %15.12f) dir=(%f %f %f) "
                      "evt=%d thread=%d "
                      "e=%f safety=%f out of GPU\n",
             __LINE__, IsElectron ? -11 : 11, lvolID, pos[0], pos[1], pos[2], dir[0], dir[1], dir[2],
             currentTrack.eventId, currentTrack.threadId, currentTrack.eKin,
             BVHNavigator::ComputeSafety(pos, navState));
      // pos += kPushOutRegion * dir;
      // survive(true);
      // continue;
    }

    if (currentTrack.stepCounter >= maxSteps) {
      printf("Killing e-/+ event %d E=%f vol=%d lvol=%d after %d steps.\n", currentTrack.eventId, eKin, volume->id(),
             lvolID, currentTrack.stepCounter);
      continue;
    }

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack elTrack;
    G4HepEmTrack *theTrack = elTrack.GetTrack();
    theTrack->SetEKin(eKin);
    theTrack->SetMCIndex(auxData.fMCIndex);
    theTrack->SetOnBoundary(navState.IsOnBoundary());
    theTrack->SetCharge(Charge);
    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    mscData->fIsFirstStep        = currentTrack.initialRange < 0;
    mscData->fInitialRange       = currentTrack.initialRange;
    mscData->fDynamicRangeFactor = currentTrack.dynamicRangeFactor;
    mscData->fTlimitMin          = currentTrack.tlimitMin;

    // Prepare a branched RNG state while threads are synchronized. Even if not
    // used, this provides a fresh round of random numbers and reduces thread
    // divergence because the RNG state doesn't need to be advanced later.
    RanluxppDouble newRNG(currentTrack.rngState.BranchNoAdvance());

    // Compute safety, needed for MSC step limit.
    double safety = 0;
    if (!navState.IsOnBoundary()) {
      safety = BVHNavigator::ComputeSafety(pos, navState);
    }
    theTrack->SetSafety(safety);

    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 4; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft = -std::log(currentTrack.Uniform());
      }
      if (ip == 3) numIALeft = vecgeom::kInfLength; // suppress lepton nuclear by infinite length
      theTrack->SetNumIALeft(numIALeft, ip);
    }

    G4HepEmElectronManager::HowFarToDiscreteInteraction(&g4HepEmData, &g4HepEmPars, &elTrack);

    bool restrictedPhysicalStepLength = false;
    if (BzFieldValue != 0) {
      const double momentumMag = sqrt(eKin * (eKin + 2.0 * restMass));
      // Distance along the track direction to reach the maximum allowed error
      const double safeLength = fieldPropagatorBz.ComputeSafeLength(momentumMag, Charge, dir);

      constexpr int MaxSafeLength = 10;
      double limit                = MaxSafeLength * safeLength;
      limit                       = safety > limit ? safety : limit;

      double physicalStepLength = elTrack.GetPStepLength();
      if (physicalStepLength > limit) {
        physicalStepLength           = limit;
        restrictedPhysicalStepLength = true;
        elTrack.SetPStepLength(physicalStepLength);
        // Note: We are limiting the true step length, which is converted to
        // a shorter geometry step length in HowFarToMSC. In that sense, the
        // limit is an over-approximation, but that is fine for our purpose.
      }
    }

    G4HepEmElectronManager::HowFarToMSC(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    // Remember MSC values for the next step(s).
    currentTrack.initialRange       = mscData->fInitialRange;
    currentTrack.dynamicRangeFactor = mscData->fDynamicRangeFactor;
    currentTrack.tlimitMin          = mscData->fTlimitMin;

    // Get result into variables.
    const double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();
    // The phyiscal step length is the amount that the particle experiences
    // which might be longer than the geometrical step length due to MSC. As
    // long as we call PerformContinuous in the same kernel we don't need to
    // care, but we need to make this available when splitting the operations.
    // double physicalStepLength = elTrack.GetPStepLength();
    const int winnerProcessIndex = theTrack->GetWinnerProcessIndex();
    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    bool propagated    = true;
    long hitsurf_index = -1;
    double geometryStepLength;
    vecgeom::NavigationState nextState;
    if (BzFieldValue != 0) {
      geometryStepLength = fieldPropagatorBz.ComputeStepAndNextVolume<BVHNavigator>(
          eKin, restMass, Charge, geometricalStepLengthFromPhysics, pos, dir, navState, nextState, hitsurf_index,
          propagated, safety, /*max_i_in_stepper=*/10);
    } else {
      geometryStepLength = BVHNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState,
                                                                  nextState, kPush);
      pos += geometryStepLength * dir;
    }

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to MSC.
    theTrack->SetDirection(dir.x(), dir.y(), dir.z());
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    // Apply continuous effects.
    bool stopped = G4HepEmElectronManager::PerformContinuous(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    // Collect the direction change and displacement by MSC.
    const double *direction = theTrack->GetDirection();
    dir.Set(direction[0], direction[1], direction[2]);
    if (!nextState.IsOnBoundary()) {
      const double *mscDisplacement = mscData->GetDisplacement();
      vecgeom::Vector3D<Precision> displacement(mscDisplacement[0], mscDisplacement[1], mscDisplacement[2]);
      const double dLength2            = displacement.Length2();
      constexpr double kGeomMinLength  = 5 * copcore::units::nm;          // 0.05 [nm]
      constexpr double kGeomMinLength2 = kGeomMinLength * kGeomMinLength; // (0.05 [nm])^2
      if (dLength2 > kGeomMinLength2) {
        const double dispR = std::sqrt(dLength2);
        // Estimate safety by subtracting the geometrical step length.
        safety -= geometryStepLength;
        constexpr double sFact = 0.99;
        double reducedSafety   = sFact * safety;

        // Apply displacement, depending on how close we are to a boundary.
        // 1a. Far away from geometry boundary:
        if (reducedSafety > 0.0 && dispR <= reducedSafety) {
          pos += displacement;
        } else {
          // Recompute safety.
          safety        = BVHNavigator::ComputeSafety(pos, navState);
          reducedSafety = sFact * safety;

          // 1b. Far away from geometry boundary:
          if (reducedSafety > 0.0 && dispR <= reducedSafety) {
            pos += displacement;
            // 2. Push to boundary:
          } else if (reducedSafety > kGeomMinLength) {
            pos += displacement * (reducedSafety / dispR);
          }
          // 3. Very small safety: do nothing.
        }
      }
    }

    // Collect the charged step length (might be changed by MSC). Collect the changes in energy and deposit.
    eKin                       = theTrack->GetEKin();
    const double energyDeposit = theTrack->GetEnergyDeposit();

    // Update the flight times of the particle
    // By calculating the velocity here, we assume that all the energy deposit is done at the PreStepPoint, and
    // the velocity depends on the remaining energy
    const double deltaTime = elTrack.GetPStepLength() / GetVelocity(eKin);
    currentTrack.globalTime += deltaTime;
    currentTrack.localTime += deltaTime;
    currentTrack.properTime += deltaTime * (restMass / eKin);

    if (nextState.IsOnBoundary()) {
      // if the particle hit a boundary, and is neither stopped or outside, relocate to have the correct next state
      // before RecordHit is called
      if (!stopped && !nextState.IsOutside()) {
#ifdef ADEPT_USE_SURF
        AdePTNavigator::RelocateToNextVolume(pos, dir, hitsurf_index, nextState);
#else
        AdePTNavigator::RelocateToNextVolume(pos, dir, nextState);
#endif
      }
    }

    if (auxData.fSensIndex >= 0)
      adept_scoring::RecordHit(&userScoring[currentTrack.threadId], currentTrack.parentId,
                               static_cast<char>(IsElectron ? 0 : 1), // Particle type
                               elTrack.GetPStepLength(),              // Step length
                               energyDeposit,                         // Total Edep
                               &navState,                             // Pre-step point navstate
                               &preStepPos,                           // Pre-step point position
                               &preStepDir,                           // Pre-step point momentum direction
                               nullptr,                               // Pre-step point polarization
                               preStepEnergy,                         // Pre-step point kinetic energy
                               &nextState,                            // Post-step point navstate
                               &pos,                                  // Post-step point position
                               &dir,                                  // Post-step point momentum direction
                               nullptr,                               // Post-step point polarization
                               eKin,                                  // Post-step point kinetic energy
                               currentTrack.eventId, currentTrack.threadId);

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 4; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
    }

    if (stopped) {
      if (!IsElectron) {
        // Annihilate the stopped positron into two gammas heading to opposite
        // directions (isotropic).
        adept_scoring::AccountProduced(userScoring + currentTrack.threadId, /*numElectrons*/ 0, /*numPositrons*/ 0,
                                       /*numGammas*/ 2);

        const double cost = 2 * currentTrack.Uniform() - 1;
        const double sint = sqrt(1 - cost * cost);
        const double phi  = k2Pi * currentTrack.Uniform();
        double sinPhi, cosPhi;
        sincos(phi, &sinPhi, &cosPhi);

        newRNG.Advance();
        Track &gamma1 = secondaries.gammas.NextTrack(newRNG, double{copcore::units::kElectronMassC2}, pos,
                                                     vecgeom::Vector3D<Precision>{sint * cosPhi, sint * sinPhi, cost},
                                                     navState, currentTrack);

        // Reuse the RNG state of the dying track.
        Track &gamma2 = secondaries.gammas.NextTrack(currentTrack.rngState, double{copcore::units::kElectronMassC2},
                                                     pos, -gamma1.dir, navState, currentTrack);
      }
      // Particles are killed by not enqueuing them into the new activeQueue.
      slotManager.MarkSlotForFreeing(slot);
      continue;
    }

    if (nextState.IsOnBoundary()) {
      // For now, just count that we hit something.

      // Kill the particle if it left the world.
      if (!nextState.IsOutside()) {

        // Move to the next boundary.
        navState = nextState;
        // Check if the next volume belongs to the GPU region and push it to the appropriate queue
#ifndef ADEPT_USE_SURF
        const int nextlvolID = navState.Top()->GetLogicalVolume()->id();
#else
        const int nextlvolID = navState.GetLogicalId();
#endif
        VolAuxData const &nextauxData = AsyncAdePT::gVolAuxData[nextlvolID];
        if (nextauxData.fGPUregion > 0)
          survive();
        else {
          // To be safe, just push a bit the track exiting the GPU region to make sure
          // Geant4 does not relocate it again inside the same region
          pos += kPushOutRegion * dir;
          survive(/*leak*/ true);
        }
      } else {
        // Kill the particle if it left the world.
        slotManager.MarkSlotForFreeing(slot);
      }
      continue;
    } else if (!propagated || restrictedPhysicalStepLength) {
      // Did not yet reach the interaction point due to error in the magnetic
      // field propagation. Try again next time.
      survive();
      continue;
    } else if (winnerProcessIndex < 0) {
      // No discrete process, move on.
      survive();
      continue;
    }

    // if (nextState.IsOnBoundary()) {
    //   // TODO: We need to return these particles to G4, also check if we
    //   // want to have this logic here
    //   if (++currentTrack.looperCounter > 256) {
    //     // Kill loopers that are scraping a boundary
    //     printf("Looper scraping a boundary going to G4: E=%E event=%d loop=%d energyDeposit=%E geoStepLength=%E "
    //            "physicsStepLength=%E "
    //            "safety=%E\n",
    //            eKin, currentTrack.eventId, currentTrack.looperCounter, energyDeposit, geometryStepLength,
    //            geometricalStepLengthFromPhysics, safety);
    //     atomicAdd(&userScoring[currentTrack.threadId].fGlobalCounters.numKilled, 1);
    //     // survive(/*leak=*/true); // don't send particles back at this point, G4 seems to be struggling a lot with
    //     // those tracks too!
    //     continue;
    //   } else if (nextState.Top() != nullptr) {
    //     // Go to next volume
    //     BVHNavigator::RelocateToNextVolume(pos, dir, nextState);

    //     // Move to the next boundary.
    //     navState                   = nextState;
    //     currentTrack.looperCounter = 0;
    //     // Check if the next volume belongs to the GPU region and push it to the appropriate queue
    //     const auto nextvolume         = navState.Top();
    //     const int nextlvolID          = nextvolume->GetLogicalVolume()->id();
    //     VolAuxData const &nextauxData = AsyncAdePT::gVolAuxData[nextlvolID];
    //     if (nextauxData.fGPUregion > 0)
    //       survive();
    //     else {
    //       // To be safe, just push a bit the track exiting the GPU region to make sure
    //       // Geant4 does not relocate it again inside the same region
    //       pos += kPushOutRegion * dir;
    //       survive(/*leak*/ true);
    //     }
    //   } else {
    //     // Kill the particle if it left the world.
    //     slotManager.MarkSlotForFreeing(slot);
    //   }
    //   continue;
    // } else if (!propagated || restrictedPhysicalStepLength) {
    //   // Did not yet reach the interaction point due to error in the magnetic
    //   // field propagation. Try again next time.
    //   if (++currentTrack.looperCounter > 1024) {
    //     // Kill loopers that are failing to propagate
    //     printf("Looper with trouble in field propagation going to G4: E=%E event=%d loop=%d energyDeposit=%E "
    //            "geoStepLength=%E physicsStepLength=%E "
    //            "safety=%E\n",
    //            eKin, currentTrack.eventId, currentTrack.looperCounter, energyDeposit, geometryStepLength,
    //            geometricalStepLengthFromPhysics, safety);
    //     atomicAdd(&userScoring[currentTrack.threadId].fGlobalCounters.numKilled, 1);
    //     // survive(/*leak=*/true);  // don't send particles back at this point, G4 seems to be struggling a lot with
    //     // those tracks too!

    //   } else {
    //     survive();
    //   }
    //   continue;
    // } else if (winnerProcessIndex < 0) {
    //   // No discrete process, move on.
    //   currentTrack.looperCounter = 0;
    //   survive();
    //   continue;
    // }
    // currentTrack.looperCounter = 0;

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    currentTrack.numIALeft[winnerProcessIndex] = -1.0;

    // Check if a delta interaction happens instead of the real discrete process.
    if (G4HepEmElectronManager::CheckDelta(&g4HepEmData, theTrack, currentTrack.Uniform())) {
      // A delta interaction happened, move on.
      survive();
      continue;
    }

    // Perform the discrete interaction, make sure the branched RNG state is
    // ready to be used.
    newRNG.Advance();
    // Also advance the current RNG state to provide a fresh round of random
    // numbers after MSC used up a fair share for sampling the displacement.
    currentTrack.rngState.Advance();

    const double theElCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;

    switch (winnerProcessIndex) {
    case 0: {
      // Invoke ionization (for e-/e+):
      double deltaEkin = (IsElectron) ? G4HepEmElectronInteractionIoni::SampleETransferMoller(theElCut, eKin, &rnge)
                                      : G4HepEmElectronInteractionIoni::SampleETransferBhabha(theElCut, eKin, &rnge);

      double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirSecondary[3];
      G4HepEmElectronInteractionIoni::SampleDirections(eKin, deltaEkin, dirSecondary, dirPrimary, &rnge);

      Track &secondary = secondaries.electrons.NextTrack(
          newRNG, deltaEkin, pos, vecgeom::Vector3D<Precision>{dirSecondary[0], dirSecondary[1], dirSecondary[2]},
          navState, currentTrack);

      adept_scoring::AccountProduced(userScoring + currentTrack.threadId, /*numElectrons*/ 1, /*numPositrons*/ 0,
                                     /*numGammas*/ 0);

      eKin -= deltaEkin;
      dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      survive();
      break;
    }
    case 1: {
      // Invoke model for Bremsstrahlung: either SB- or Rel-Brem.
      double logEnergy = std::log(eKin);
      double deltaEkin = eKin < g4HepEmPars.fElectronBremModelLim
                             ? G4HepEmElectronInteractionBrem::SampleETransferSB(&g4HepEmData, eKin, logEnergy,
                                                                                 auxData.fMCIndex, &rnge, IsElectron)
                             : G4HepEmElectronInteractionBrem::SampleETransferRB(&g4HepEmData, eKin, logEnergy,
                                                                                 auxData.fMCIndex, &rnge, IsElectron);

      double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirSecondary[3];
      G4HepEmElectronInteractionBrem::SampleDirections(eKin, deltaEkin, dirSecondary, dirPrimary, &rnge);

      adept_scoring::AccountProduced(userScoring + currentTrack.threadId, /*numElectrons*/ 0, /*numPositrons*/ 0,
                                     /*numGammas*/ 1);

      secondaries.gammas.NextTrack(newRNG, deltaEkin, pos,
                                   vecgeom::Vector3D<Precision>{dirSecondary[0], dirSecondary[1], dirSecondary[2]},
                                   navState, currentTrack);

      eKin -= deltaEkin;
      dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      survive();
      break;
    }
    case 2: {
      // Invoke annihilation (in-flight) for e+
      double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double theGamma1Ekin, theGamma2Ekin;
      double theGamma1Dir[3], theGamma2Dir[3];
      G4HepEmPositronInteractionAnnihilation::SampleEnergyAndDirectionsInFlight(
          eKin, dirPrimary, &theGamma1Ekin, theGamma1Dir, &theGamma2Ekin, theGamma2Dir, &rnge);

      adept_scoring::AccountProduced(userScoring + currentTrack.threadId, /*numElectrons*/ 0, /*numPositrons*/ 0,
                                     /*numGammas*/ 2);

      secondaries.gammas.NextTrack(newRNG, theGamma1Ekin, pos,
                                   vecgeom::Vector3D<Precision>{theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]},
                                   navState, currentTrack);
      // Reuse the RNG state of the dying track.
      secondaries.gammas.NextTrack(currentTrack.rngState, theGamma2Ekin, pos,
                                   vecgeom::Vector3D<Precision>{theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]},
                                   navState, currentTrack);

      // The current track is killed by not enqueuing into the next activeQueue.
      slotManager.MarkSlotForFreeing(slot);
      break;
    }
    case 3: {
      // leptop-nuclear needs to be handled by G4, just keep the track
      survive();
      break;
    }
    }
  }
}

// Instantiate kernels for electrons and positrons.
template <typename Scoring>
__global__ void TransportElectrons(Track *electrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, adept::MParray *leakedQueue, Scoring *userScoring)
{
  TransportElectrons</*IsElectron*/ true, Scoring>(electrons, active, secondaries, activeQueue, leakedQueue,
                                                   userScoring);
}
template <typename Scoring>
__global__ void TransportPositrons(Track *positrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, adept::MParray *leakedQueue, Scoring *userScoring)
{
  TransportElectrons</*IsElectron*/ false, Scoring>(positrons, active, secondaries, activeQueue, leakedQueue,
                                                    userScoring);
}

} // namespace AsyncAdePT
