import type { PlayingSurface } from '../../enums/PlayingSurface.js';
import type { Venue } from '../../enums/Venue.js';

export interface GameTable {
   baseball_reference_id: string,
   day_of_year: number,
   game_id: number,
   game_of_season: number,
   home_plate_umpire: number,
   host_moneyline: number | null,
   host_score: number,
   host_team_id: number,
   hour_of_day: number,
   over_moneyline: number | null,
   over_under: number | null,
   playing_surface: PlayingSurface,
   season: number,
   temperature: number,
   under_moneyline: number | null,
   venue: Venue,
   visitor_moneyline: number | null,
   visitor_score: number,
   visitor_team_id: number,
}