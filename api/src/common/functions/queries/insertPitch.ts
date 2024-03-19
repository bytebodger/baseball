import type { Pitch } from '../../enums/Pitch.js';
import { Table } from '../../enums/Table.js';
import { insertIntoTable } from './insertIntoTable.js';

interface Fields {
   at_bat_id: number,
   result: Pitch,
   sequence_id: number,
}

export const insertPitch = async (fields: Fields) => {
   return await insertIntoTable(Table.pitch, fields);
}