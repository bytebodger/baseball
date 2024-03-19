import { Table } from '../../enums/Table.js';
import { insertIntoTable } from './insertIntoTable.js';

interface Fields {
   bases: number,
   batter_player_id: number,
   game_id: number,
   outs: number,
   pitcher_player_id: number,
   result: number,
   runs: number,
   sequence_id: number,
   total_pitches: number,
}

export const insertAtBat = async (fields: Fields) => {
   return await insertIntoTable(Table.atBat, fields);
}