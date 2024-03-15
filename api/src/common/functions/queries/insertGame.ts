import type { PlayingSurface } from '../../enums/PlayingSurface.js';
import { Table } from '../../enums/Table.js';
import type { Umpire } from '../../enums/Umpire.js';
import type { Venue } from '../../enums/Venue.js';
import { insertIntoTable } from './insertIntoTable.js';

interface Fields {
   baseball_reference_id: string,
   day_of_year: number,
   game_of_season: number,
   home_plate_umpire: Umpire,
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

export const insertGame = async (fields: Fields) => {
   return await insertIntoTable(Table.game, fields);
}