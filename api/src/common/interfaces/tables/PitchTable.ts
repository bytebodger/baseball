import type { Pitch } from '../../enums/Pitch.js';

export interface PitchTable {
   at_bat_id: number,
   pitch_id: number,
   result: Pitch,
   sequence_id: number,
}