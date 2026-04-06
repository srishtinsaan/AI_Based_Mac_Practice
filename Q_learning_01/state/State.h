#ifndef STATE_H
#define STATE_H


enum MacTableStatus {
    EMPTY = 0,
    ALMOST_FULL = 1,
    FULL = 2
};

enum FloodPressure {
    LOW_FP = 0,
    MEDIUM_FP = 1,
    HIGH_FP = 2
};

enum PortTraffic {
    LOW_TRAFFIC = 0,
    MEDIUM_TRAFFIC = 1,
    HIGH_TRAFFIC = 2
};

enum EntryAge {
    FRESH = 0,
    AGING = 1,
    STALE = 2
};

enum NewMacRate {
    LOW_RATE = 0,
    HIGH_RATE = 1
};

struct State {

    MacTableStatus macTableStatus;

    FloodPressure floodPressure;

    PortTraffic portTraffic;

    int VLANDevices;

    EntryAge entryAge;

    NewMacRate newMacRate;

    State() {
        macTableStatus = EMPTY;
        floodPressure = LOW_FP;
        portTraffic = LOW_TRAFFIC;
        VLANDevices = 0;
        entryAge = FRESH;
        newMacRate = LOW_RATE;
    }
};

#endif