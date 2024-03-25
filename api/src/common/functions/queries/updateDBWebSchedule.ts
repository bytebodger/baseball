import { IdentityField } from '../../enums/IdentityField.js';
import { Table } from '../../enums/Table.js';
import { updateDBTableByPrimaryKey } from './updateDBTableByPrimaryKey.js';

interface Fields {
   has_been_played?: boolean,
   html?: string,
   time_checked: number,
   time_processed?: number | null,
   time_retrieved?: number,
   web_schedule_id: number,
}

export const updateDBWebSchedule = async (fields: Fields) => {
   return await updateDBTableByPrimaryKey(Table.webSchedule, IdentityField.webSchedule, fields);
}