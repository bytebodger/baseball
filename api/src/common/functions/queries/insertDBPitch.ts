import type { Pitch } from '../../enums/Pitch.js';
import { Table } from '../../enums/Table.js';
import { insertIntoDBTable } from './insertIntoDBTable.js';

interface Fields {
   at_bat_id: number,
   result: Pitch,
   sequence_id: number,
}

export const insertDBPitch = async (fields: Fields) => {
   return await insertIntoDBTable(Table.pitch, fields);
}